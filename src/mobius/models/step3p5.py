# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Step-3.5-Flash text decoder.

The public checkpoint (``stepfun-ai/Step-3.5-Flash``, ``model_type=step3p5``)
is a decoder-only MoE transformer. This module exports that decoder. The
three trailing multi-token-prediction layers are not part of the graph;
their checkpoint keys are dropped in :meth:`Step3p5CausalLMModel.preprocess_weights`.

Per layer, compared with a stock Llama block:

* Gemma-style RMSNorm (``weight + 1``) on the residual stream and on Q/K.
* Full attention and sliding-window attention alternate 3:1. Sliding layers
  use a wider query head count. Full-attention layers use Llama-3 RoPE on
  half of ``head_dim``; sliding layers use unscaled RoPE on the whole head.
* A per-head sigmoid gate multiplies the attention output before ``o_proj``.
* The first layers are dense SwiGLU. The rest add a sigmoid top-k MoE
  (selection bias does not enter the weights) to an always-on shared expert.
  Some late layers clamp the SwiGLU gate and up projections.
"""

from __future__ import annotations

import dataclasses
import math

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, Step3p5Config
from mobius.components import (
    MLP,
    Embedding,
    Linear,
    MoELayer,
    OffsetRMSNorm,
    apply_rotary_pos_emb,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._attention import Attention, _apply_attention
from mobius.components._moe import _scatter_selected_to_full
from mobius.models.base import (
    CausalLMModel,
    effective_tie_word_embeddings,
    embedding_for_config,
    linear_class_for_config,
)

_LAYER_TYPES = frozenset({"full_attention", "sliding_attention"})
_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _layer_types(config: Step3p5Config) -> list[str]:
    if config.layer_types:
        kinds = list(config.layer_types)
    else:
        kinds = [
            "full_attention" if index % 4 == 0 else "sliding_attention"
            for index in range(config.num_hidden_layers)
        ]
    if len(kinds) != config.num_hidden_layers:
        raise ValueError(
            "Step3p5 layer_types must have one entry per decoder layer, "
            f"got {len(kinds)} for num_hidden_layers={config.num_hidden_layers}."
        )
    unknown = sorted(set(kinds) - _LAYER_TYPES)
    if unknown:
        raise ValueError(f"Step3p5 only supports {_LAYER_TYPES}, got {unknown}.")
    if "sliding_attention" in kinds and (
        config.sliding_window is None or config.sliding_window <= 0
    ):
        raise ValueError("Step3p5 sliding-attention layers require a positive sliding_window.")
    return kinds


def _is_moe_layer(config: Step3p5Config, layer_idx: int) -> bool:
    if config.moe_layer_indices is not None:
        return layer_idx in config.moe_layer_indices
    return layer_idx >= config.first_k_dense_replace


def _swiglu_limit(limits: list[float] | None, layer_idx: int) -> float | None:
    """Return the clamp bound, treating ``0`` and a missing entry as no clamp."""
    if limits is None or layer_idx >= len(limits):
        return None
    value = float(limits[layer_idx])
    if math.isclose(value, 0.0):
        return None
    return value


def _decoder_layer_index(key: str) -> int | None:
    marker = ".layers."
    start = key.find(marker)
    if start < 0:
        return None
    number = key[start + len(marker) :].split(".", 1)[0]
    if number.isdigit():
        return int(number)
    return None


class _ClampedMLP(MLP):
    """SwiGLU MLP with the optional Step3p5 gate/up clamp."""

    def __init__(
        self,
        config: ArchitectureConfig,
        limit: float | None,
        linear_class: type | None = None,
    ):
        super().__init__(config, linear_class=linear_class)
        self._limit = limit

    def forward(self, op: OpBuilder, x: ir.Value):
        gate = self.act_fn(op, self.gate_proj(op, x))
        up = self.up_proj(op, x)
        if self._limit is not None:
            limit = op.CastLike(self._limit, gate)
            gate = op.Min(gate, limit)
            up = op.Min(op.Max(up, op.Neg(limit)), limit)
        return self.down_proj(op, op.Mul(gate, up))


class Step3p5TopKGate(nn.Module):
    """Sigmoid top-k router. The bias changes which experts are chosen only."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        routed_scaling_factor: float,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.weight = nn.Parameter([num_experts, hidden_size])
        self.weight._keep_float32 = True  # type: ignore[attr-defined]

    def forward(self, op: OpBuilder, hidden_states: ir.Value, router_bias: ir.Value):
        hidden_fp32 = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        weight_fp32 = op.Cast(self.weight, to=ir.DataType.FLOAT)
        logits = op.MatMul(hidden_fp32, op.Transpose(weight_fp32, perm=[1, 0]))
        probabilities = op.Sigmoid(logits)
        choice = op.Add(probabilities, op.Cast(router_bias, to=ir.DataType.FLOAT))
        _scores, selected = op.TopK(
            choice,
            op.Constant(value_ints=[self.top_k]),
            axis=-1,
            largest=1,
            sorted=1,
            _outputs=2,
        )
        weights = op.GatherElements(probabilities, selected, axis=-1)
        denominator = op.Add(
            op.ReduceSum(weights, [-1], keepdims=True),
            op.CastLike(1e-20, weights),
        )
        weights = op.Div(weights, denominator)
        if self.routed_scaling_factor != 1.0:  # noqa: RUF069
            weights = op.Mul(weights, op.CastLike(self.routed_scaling_factor, weights))
        return weights, selected

    def qmoe_routing(self, op: OpBuilder, hidden_states: ir.Value, router_bias: ir.Value):
        """Encode the already-final routing pair for ``com.microsoft.QMoE``.

        Selection uses ``sigmoid(logits) + bias`` while the gathered weights
        come from the unbiased sigmoid, so QMoE cannot recompute them from
        logits. The scatter adapter marks the chosen experts and stores the
        final weights beside them. ``normalize_routing_weights`` stays off.
        CPU QMoE honors that pair. CUDA QMoE ignores the separate weight
        input and is not a match for this router.
        """
        weights, selected = self.forward(op, hidden_states, router_bias)
        weights = op.CastLike(weights, hidden_states)
        router_probs, router_weights = _scatter_selected_to_full(
            op, weights, selected, self.num_experts
        )
        return router_probs, router_weights, False, 1.0


class Step3p5MoE(MoELayer):
    """Sigmoid-routed SwiGLU experts. The shared expert lives on the layer.

    An INT4 config that matches the QMoE ABI packs the routed experts into one
    ``com.microsoft.QMoE`` node. Otherwise every expert is a separate MLP.

    QMoE clamps the gate before SiLU, while the reference clamps after SiLU.
    Below the limit the two agree. Above it QMoE returns ``silu(limit)``
    instead of ``limit``, which is within 0.1% for the checkpoint's limit of 7.
    """

    def __init__(
        self,
        config: Step3p5Config,
        *,
        routed_limit: float | None,
        linear_class: type | None,
    ):
        if config.num_local_experts is None or config.num_experts_per_tok is None:
            raise ValueError(
                "Step3p5 MoE layers require num_local_experts and num_experts_per_tok."
            )
        if config.moe_intermediate_size is None:
            raise ValueError("Step3p5 MoE layers require moe_intermediate_size.")
        gate = Step3p5TopKGate(
            config.hidden_size,
            config.num_local_experts,
            config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
        )
        super().__init__(
            config,
            gate=gate,
            linear_class=linear_class,
            expert_factory=lambda cfg, lc: _ClampedMLP(cfg, routed_limit, linear_class=lc),
            activation_alpha=1.0,
            activation_beta=0.0,
            swiglu_limit=math.inf if routed_limit is None else routed_limit,
        )
        self.router_bias = nn.Parameter(
            [config.num_local_experts],
            dtype=ir.DataType.FLOAT,
        )
        self.router_bias._keep_float32 = True  # type: ignore[attr-defined]

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        if self.experts is None:
            return self._qmoe_forward(op, hidden_states, self.router_bias)
        weights, selected = self.gate(op, hidden_states, self.router_bias)
        routed = None
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(op, hidden_states)
            match = op.Equal(selected, op.Constant(value_int=expert_idx))
            match_float = op.CastLike(match, weights)
            weight = op.ReduceSum(op.Mul(weights, match_float), [-1], keepdims=True)
            contribution = op.CastLike(
                op.Mul(op.Cast(expert_output, to=ir.DataType.FLOAT), weight),
                hidden_states,
            )
            routed = contribution if routed is None else op.Add(routed, contribution)
        assert routed is not None
        return routed


class Step3p5Attention(Attention):
    """GQA with offset Q/K norm, partial RoPE, and a per-head output gate."""

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__(
            config,
            rms_norm_class=OffsetRMSNorm,
            linear_class=linear_class,
        )
        self.g_proj = None
        if getattr(config, "use_head_wise_attn_gate", True):
            factory = linear_class or Linear
            self.g_proj = factory(config.hidden_size, self.num_attention_heads, bias=False)

    def _apply_head_gate(self, op: OpBuilder, attn_output: ir.Value, hidden_states: ir.Value):
        assert self.g_proj is not None
        gate = op.Unsqueeze(op.Sigmoid(self.g_proj(op, hidden_states)), [-1])
        shaped = op.Reshape(attn_output, [0, 0, self.num_attention_heads, self.head_dim])
        gated = op.Mul(shaped, gate)
        return op.Reshape(gated, [0, 0, self.num_attention_heads * self.head_dim])

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
    ):
        query_states, key_states, value_states = self._project_qkv(op, hidden_states)
        if self.q_norm is not None and self.k_norm is not None:
            query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
            key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
            query_states = self.q_norm(op, query_states)
            key_states = self.k_norm(op, key_states)
            query_states = op.Reshape(query_states, [0, 0, -1])
            key_states = op.Reshape(key_states, [0, 0, -1])

        if position_embeddings is not None:
            query_states = apply_rotary_pos_emb(
                op,
                x=query_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )
            key_states = apply_rotary_pos_emb(
                op,
                x=key_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_key_value_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )

        attn_output, present_key, present_value = _apply_attention(
            op,
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling,
        )
        if self.g_proj is not None:
            attn_output = self._apply_head_gate(op, attn_output, hidden_states)
        return self._project_output(op, attn_output), (present_key, present_value)


class Step3p5DecoderLayer(nn.Module):
    """One pre-norm Step3p5 block, dense or routed."""

    def __init__(self, config: Step3p5Config, layer_idx: int, layer_type: str):
        super().__init__()
        linear_class = linear_class_for_config(config)
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        partial = config.full_partial_rotary_factor
        if layer_type == "sliding_attention":
            if config.sliding_num_attention_heads is not None:
                heads = config.sliding_num_attention_heads
            if config.sliding_num_key_value_heads is not None:
                kv_heads = config.sliding_num_key_value_heads
            partial = config.sliding_partial_rotary_factor
        if heads % kv_heads != 0:
            raise ValueError(
                f"Step3p5 layer {layer_idx} has {heads} query heads and {kv_heads} "
                "KV heads; the query count must be a multiple of the KV count."
            )
        attn_config = dataclasses.replace(
            config,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            partial_rotary_factor=partial,
            attn_qk_norm=True,
            attn_qk_norm_full=False,
            attn_qkv_bias=False,
            attn_o_bias=False,
        )
        self.self_attn = Step3p5Attention(attn_config, linear_class=linear_class)
        shared_limit = _swiglu_limit(config.swiglu_limits_shared, layer_idx)
        if _is_moe_layer(config, layer_idx):
            if config.shared_expert_intermediate_size is None:
                raise ValueError("Step3p5 MoE layers require shared_expert_intermediate_size.")
            self.moe = Step3p5MoE(
                config,
                routed_limit=_swiglu_limit(config.swiglu_limits, layer_idx),
                linear_class=linear_class,
            )
            shared_config = dataclasses.replace(
                config, intermediate_size=config.shared_expert_intermediate_size
            )
            self.share_expert = _ClampedMLP(
                shared_config, shared_limit, linear_class=linear_class
            )
            self.mlp = None
        else:
            self.moe = None
            self.share_expert = None
            self.mlp = _ClampedMLP(
                config,
                _swiglu_limit(config.swiglu_limits_shared, layer_idx),
                linear_class=linear_class,
            )
        self.input_layernorm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_out, present_kv = self.self_attn(
            op,
            hidden_states,
            attention_bias,
            position_embeddings,
            past_key_value,
        )
        hidden_states = op.Add(residual, attn_out)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        if self.moe is not None:
            assert self.share_expert is not None
            hidden_states = op.Add(
                self.share_expert(op, hidden_states), self.moe(op, hidden_states)
            )
        else:
            assert self.mlp is not None
            hidden_states = self.mlp(op, hidden_states)
        return op.Add(residual, hidden_states), present_kv


def _rope_module(config: Step3p5Config, *, llama3: bool, theta: float, partial: float):
    rope_config = dataclasses.replace(
        config,
        rope_type="llama3" if llama3 else "default",
        rope_theta=theta,
        partial_rotary_factor=partial,
        rope_scaling=config.rope_scaling if llama3 else None,
    )
    rotary = initialize_rope(rope_config)
    if rotary is None:
        raise ValueError("Step3p5 requires rotary position embeddings.")
    return rotary


class Step3p5TextModel(nn.Module):
    """Step3p5 token backbone with two RoPE tables and two attention masks."""

    def __init__(self, config: Step3p5Config):
        super().__init__()
        self.config = config
        self._layer_types = _layer_types(config)
        self._sliding_window = config.sliding_window
        self.embed_tokens = embedding_for_config(config)
        self.layers = nn.ModuleList(
            [
                Step3p5DecoderLayer(config, index, kind)
                for index, kind in enumerate(self._layer_types)
            ]
        )
        self.norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        full_theta = (
            config.rope_theta if config.full_rope_theta is None else config.full_rope_theta
        )
        use_llama3 = bool(config.yarn_on_full_attention and config.rope_scaling)
        if "full_attention" in self._layer_types:
            self.full_rotary_emb = _rope_module(
                config,
                llama3=use_llama3,
                theta=full_theta,
                partial=config.full_partial_rotary_factor,
            )
        else:
            self.full_rotary_emb = None
        if "sliding_attention" in self._layer_types:
            self.sliding_rotary_emb = _rope_module(
                config,
                llama3=False,
                theta=config.sliding_rope_theta,
                partial=config.sliding_partial_rotary_factor,
            )
        else:
            self.sliding_rotary_emb = None

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        embeddings: dict[str, tuple] = {}
        if self.full_rotary_emb is not None:
            embeddings["full_attention"] = self.full_rotary_emb(op, position_ids)
        if self.sliding_rotary_emb is not None:
            embeddings["sliding_attention"] = self.sliding_rotary_emb(op, position_ids)

        full_bias = None
        sliding_bias = None
        if attention_mask is not None:
            full_bias = create_attention_bias(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                dtype=self.config.dtype,
            )
            if self._sliding_window is not None:
                sliding_bias = create_attention_bias(
                    op,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sliding_window=self._sliding_window,
                    dtype=self.config.dtype,
                )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for index, (layer, past_kv) in enumerate(zip(self.layers, past_kvs)):
            kind = self._layer_types[index]
            bias = sliding_bias if kind == "sliding_attention" else full_bias
            hidden_states, present_kv = layer(
                op,
                hidden_states,
                bias,
                embeddings[kind],
                past_kv,
            )
            present_key_values.append(present_kv)
        return self.norm(op, hidden_states), present_key_values


def _split_expert_bank(
    key: str,
    value: torch.Tensor,
    num_experts: int | None,
) -> dict[str, torch.Tensor] | None:
    """Split ``moe.{gate,up,down}_proj`` banks into per-expert matrices."""
    matched = None
    for projection in _EXPERT_PROJECTIONS:
        suffix = f".moe.{projection}.weight"
        if key.endswith(suffix):
            matched = projection
            break
    if matched is None or value.ndim != 3:
        return None
    if num_experts is None or value.shape[0] != num_experts:
        raise ValueError(
            f"{key} has leading dimension {tuple(value.shape)}, expected {num_experts} experts."
        )
    prefix = key[: -len(f"{matched}.weight")]
    return {
        f"{prefix}experts.{expert}.{matched}.weight": value[expert]
        for expert in range(num_experts)
    }


_QMOE_FIELDS = {
    "weight": ("fc1_experts_weights", "fc2_experts_weights"),
    "scales": ("fc1_scales", "fc2_scales"),
    "zero_points": ("fc1_experts_zero_points", "fc2_experts_zero_points"),
}


def _packed_bank_field(key: str, value: torch.Tensor) -> tuple[str, str, str] | None:
    for projection in _EXPERT_PROJECTIONS:
        for field in _QMOE_FIELDS:
            suffix = f".moe.{projection}.{field}"
            if not key.endswith(suffix):
                continue
            if field == "weight" and (value.dtype != torch.uint8 or value.ndim != 4):
                return None
            return key[: -len(suffix)] + ".moe.", field, projection
    return None


def _pack_qmoe_banks(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Bind packed ``moe.{gate,up,down}_proj`` banks to QMoE parameters.

    Expects the names produced by Olive preprocessing, where each bank weight
    is ``[experts, rows, blocks, blob]``. Gate rows are followed by up rows in
    ``fc1``; ``MoELayer`` interleaves them in the graph.
    """
    banks: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        matched = _packed_bank_field(key, value)
        if matched is None:
            result[key] = value
            continue
        prefix, field, projection = matched
        banks.setdefault((prefix, field), {})[projection] = value
    for (prefix, field), by_projection in banks.items():
        missing = [name for name in _EXPERT_PROJECTIONS if name not in by_projection]
        if missing:
            raise ValueError(f"{prefix} packed expert banks are missing {missing} ({field}).")
        if field == "weight":
            by_projection = {name: value.flatten(-2) for name, value in by_projection.items()}
        fc1_name, fc2_name = _QMOE_FIELDS[field]
        gate = by_projection["gate_proj"]
        up = by_projection["up_proj"]
        if gate.ndim != 3 or gate.shape != up.shape:
            raise ValueError(
                f"{prefix} gate/up {field} banks must be matching rank-3 tensors, got "
                f"{tuple(gate.shape)} and {tuple(up.shape)}."
            )
        result[f"{prefix}{fc1_name}"] = torch.cat([gate, up], dim=1).contiguous()
        result[f"{prefix}{fc2_name}"] = by_projection["down_proj"].contiguous()
    return result


class Step3p5CausalLMModel(CausalLMModel):
    """Causal LM for Step-3.5-Flash.

    HuggingFace model type: ``step3p5``. The exported graph is the 45-layer
    decoder used for ordinary generation. Multi-token-prediction weights
    stored on later layer indices are ignored.
    """

    category: str = "Mixture of Experts"
    config_class = Step3p5Config

    def __init__(self, config: ArchitectureConfig):
        if not isinstance(config, Step3p5Config):
            raise TypeError("Step3p5CausalLMModel requires Step3p5Config.")
        nn.Module.__init__(self)
        self.config = config
        self.model = Step3p5TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if effective_tie_word_embeddings(config) and isinstance(
            self.model.embed_tokens, Embedding
        ):
            self.lm_head.weight = self.model.embed_tokens.weight

    def kv_cache_specs(self) -> list[tuple[int, int]]:
        """Per-layer KV cache shape. Sliding layers may use a different head count."""
        specs = []
        for kind in _layer_types(self.config):
            heads = self.config.num_key_value_heads
            sliding_heads = self.config.sliding_num_key_value_heads
            if kind == "sliding_attention" and sliding_heads is not None:
                heads = sliding_heads
            specs.append((heads, self.config.head_dim))
        return specs

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map the hub checkpoint onto the exported decoder.

        Drops multi-token-prediction layers and strips a ``language_model.``
        prefix. Float expert banks are split into per-expert matrices. Olive
        INT4 banks are packed into QMoE ``fc1``/``fc2`` parameters.
        """
        renamed: dict[str, torch.Tensor] = {}
        num_layers = self.config.num_hidden_layers
        num_experts = self.config.num_local_experts
        for source_name, value in state_dict.items():
            key = source_name
            if key.startswith("language_model."):
                key = key[len("language_model.") :]
            layer_idx = _decoder_layer_index(key)
            if layer_idx is not None and layer_idx >= num_layers:
                continue
            split = _split_expert_bank(key, value, num_experts)
            if split is None:
                renamed[key] = value
            else:
                renamed.update(split)
        return _pack_qmoe_banks(super().preprocess_weights(renamed))
