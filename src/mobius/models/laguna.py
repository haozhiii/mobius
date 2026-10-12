# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Poolside Laguna text decoder (``model_type=laguna``).

Laguna (XS/S/M) is a decoder-only MoE transformer. Compared with a stock
Qwen2-MoE block:

* Full-attention and sliding-window layers interleave (``layer_types``).
  Each kind has its own RoPE table: Laguna-S uses YaRN on half of
  ``head_dim`` for full layers and unscaled RoPE on the whole head for
  sliding layers. Query head counts may differ per layer.
* Per-head RMSNorm on Q and K before RoPE.
* ``softplus(g_proj(x))`` gates the attention output before ``o_proj``,
  either one gate per head or one per channel.
* Routed experts are chosen by ``sigmoid(logits) + e_score_correction_bias``
  and weighted by the unbiased sigmoid, renormalized and scaled by
  ``moe_routed_scaling_factor``. An ungated shared expert is always added.
"""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, LagunaConfig
from mobius._weight_utils import preprocess_quantized_weights
from mobius.components import (
    MLP,
    Embedding,
    Linear,
    MoELayer,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._attention import Attention
from mobius.components._moe import _scatter_selected_to_full
from mobius.models.base import (
    CausalLMModel,
    effective_tie_word_embeddings,
    embedding_for_config,
    linear_class_for_config,
)

_LAYER_TYPES = frozenset({"full_attention", "sliding_attention"})

# Laguna-S puts massive activations on the BOS (attention-sink) token: its
# routed-expert sum reaches ~7.8e4 in layer 46 and its residual ~8.6e5 in
# layer 47, past the float16 maximum of 65504. Float16 graphs therefore keep
# the residual stream and the RMSNorms in float32, and run QMoE with router
# weights divided by this power of two, multiplying back after the Cast.
_FP16_MOE_OUTPUT_SCALE = 8.0


def _float32_residual(config: ArchitectureConfig) -> bool:
    return config.dtype == ir.DataType.FLOAT16


def _layer_types(config: LagunaConfig) -> list[str]:
    kinds = list(config.layer_types or ["full_attention"] * config.num_hidden_layers)
    if len(kinds) != config.num_hidden_layers:
        raise ValueError(
            "Laguna layer_types must have one entry per decoder layer, "
            f"got {len(kinds)} for num_hidden_layers={config.num_hidden_layers}."
        )
    unknown = sorted(set(kinds) - _LAYER_TYPES)
    if unknown:
        raise ValueError(f"Laguna only supports {sorted(_LAYER_TYPES)}, got {unknown}.")
    if "sliding_attention" in kinds and not config.sliding_window:
        raise ValueError("Laguna sliding-attention layers require a positive sliding_window.")
    return kinds


def _is_moe_layer(config: LagunaConfig, layer_idx: int) -> bool:
    if not config.num_local_experts:
        return False
    return layer_idx not in set(config.dense_layer_indices or ())


def _moe_layer_paths(config: LagunaConfig) -> tuple[str, ...]:
    return tuple(
        f"model.layers.{index}.mlp"
        for index in range(config.num_hidden_layers)
        if _is_moe_layer(config, index)
    )


class LagunaAttention(Attention):
    """GQA with per-head Q/K RMSNorm, partial RoPE and a softplus output gate."""

    def __init__(
        self,
        config: ArchitectureConfig,
        gate: str | None,
        linear_class: type | None = None,
    ):
        super().__init__(config, linear_class=linear_class)
        self._gate = gate
        self._gate_source: ir.Value | None = None
        self.g_proj = None
        if gate is not None:
            width = self.num_attention_heads
            if gate == "per_element":
                width *= self.head_dim
            self.g_proj = (linear_class or Linear)(config.hidden_size, width, bias=False)

    def forward(self, op: OpBuilder, hidden_states: ir.Value, *args, **kwargs):
        self._gate_source = hidden_states
        try:
            return super().forward(op, hidden_states, *args, **kwargs)
        finally:
            self._gate_source = None

    def _project_output(self, op: OpBuilder, attn_output: ir.Value) -> ir.Value:
        if self.g_proj is not None:
            assert self._gate_source is not None
            logits = self.g_proj(op, self._gate_source)
            gate = op.CastLike(op.Softplus(op.Cast(logits, to=ir.DataType.FLOAT)), attn_output)
            if self._gate == "per_head":
                shaped = op.Reshape(
                    attn_output, [0, 0, self.num_attention_heads, self.head_dim]
                )
                gated = op.Mul(shaped, op.Unsqueeze(gate, [-1]))
                attn_output = op.Reshape(
                    gated, [0, 0, self.num_attention_heads * self.head_dim]
                )
            else:
                attn_output = op.Mul(attn_output, gate)
        return self.o_proj(op, attn_output)


class LagunaTopKGate(nn.Module):
    """Sigmoid top-k router. The correction bias changes which experts are chosen only."""

    def __init__(self, config: LagunaConfig, weight_scale: float = 1.0):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor * weight_scale
        self.softcap = config.router_logit_softcapping
        self.weight = nn.Parameter([self.num_experts, config.hidden_size])
        self.weight._keep_float32 = True  # type: ignore[attr-defined]
        self.e_score_correction_bias = nn.Parameter(
            [self.num_experts], dtype=ir.DataType.FLOAT
        )
        self.e_score_correction_bias._keep_float32 = True  # type: ignore[attr-defined]

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        """Return ``(weights, selected)``; weights are final and in the hidden dtype."""
        hidden_fp32 = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        weight_fp32 = op.Cast(self.weight, to=ir.DataType.FLOAT)
        logits = op.MatMul(hidden_fp32, op.Transpose(weight_fp32, perm=[1, 0]))
        if self.softcap > 0.0:
            cap = op.Constant(value_float=self.softcap)
            logits = op.Mul(op.Tanh(op.Div(logits, cap)), cap)
        probabilities = op.Sigmoid(logits)
        choice = op.Add(
            probabilities, op.Cast(self.e_score_correction_bias, to=ir.DataType.FLOAT)
        )
        _scores, selected = op.TopK(
            choice,
            op.Constant(value_ints=[self.top_k]),
            axis=-1,
            largest=1,
            sorted=1,
            _outputs=2,
        )
        weights = op.GatherElements(probabilities, selected, axis=-1)
        if self.norm_topk_prob:
            weights = op.Div(weights, op.ReduceSum(weights, [-1], keepdims=True))
        if self.routed_scaling_factor != 1.0:  # noqa: RUF069
            weights = op.Mul(weights, op.Constant(value_float=self.routed_scaling_factor))
        return op.CastLike(weights, hidden_states), selected

    def qmoe_routing(self, op: OpBuilder, hidden_states: ir.Value):
        """Encode the already-final routing pair for ``com.microsoft.QMoE``.

        Selection uses ``sigmoid(logits) + bias`` while the weights come from
        the unbiased sigmoid, so QMoE cannot recompute them from logits. The
        scatter adapter marks the chosen experts and carries the final weights
        in ``router_weights``; normalization and scaling are already applied.
        CPU QMoE honors that input. CUDA QMoE ignores it.
        """
        weights, selected = self.forward(op, hidden_states)
        router_probs, router_weights = _scatter_selected_to_full(
            op, weights, selected, self.num_experts
        )
        return router_probs, router_weights, False, 1.0


class LagunaSparseMoeBlock(MoELayer):
    """Sigmoid-routed SwiGLU experts plus an always-on, ungated shared expert.

    An INT4 config that matches the QMoE ABI packs the routed experts into one
    ``com.microsoft.QMoE`` node. Otherwise every expert is a separate MLP.
    Float16 graphs return the sum in float32.
    """

    def __init__(self, config: LagunaConfig, linear_class: type | None):
        if config.moe_intermediate_size is None:
            raise ValueError("Laguna MoE layers require moe_intermediate_size.")
        if config.shared_expert_intermediate_size is None:
            raise ValueError("Laguna MoE layers require shared_expert_intermediate_size.")
        self._output_scale = _FP16_MOE_OUTPUT_SCALE if _float32_residual(config) else None
        gate = LagunaTopKGate(config, 1.0 / (self._output_scale or 1.0))
        super().__init__(config, gate=gate, linear_class=linear_class)
        shared_config = dataclasses.replace(
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self.shared_expert = MLP(shared_config, linear_class=linear_class)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        routed = super().forward(op, hidden_states)
        shared = self.shared_expert(op, hidden_states)
        if self._output_scale is None:
            return op.Add(routed, shared)
        routed = op.Mul(
            op.Cast(routed, to=ir.DataType.FLOAT),
            op.Constant(value_float=self._output_scale),
        )
        return op.Add(routed, op.Cast(shared, to=ir.DataType.FLOAT))


class _Float32RMSNorm(RMSNorm):
    """RMSNorm over a float32 residual whose output is cast to the model dtype."""

    def __init__(self, hidden_size: int, eps: float, dtype: ir.DataType):
        super().__init__(hidden_size, eps=eps)
        self._dtype = dtype

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        normed = op.RMSNormalization(
            hidden_states,
            op.Cast(self.weight, to=ir.DataType.FLOAT),
            epsilon=self.variance_epsilon,
            axis=-1,
        )
        return op.Cast(normed, to=self._dtype)


def _rms_norm(config: LagunaConfig) -> RMSNorm:
    if _float32_residual(config):
        return _Float32RMSNorm(config.hidden_size, config.rms_norm_eps, config.dtype)
    return RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LagunaDecoderLayer(nn.Module):
    """One pre-norm Laguna block, dense or routed."""

    def __init__(self, config: LagunaConfig, layer_idx: int, layer_type: str):
        super().__init__()
        linear_class = linear_class_for_config(config)
        heads = config.num_attention_heads
        if config.num_attention_heads_per_layer is not None:
            heads = config.num_attention_heads_per_layer[layer_idx]
        if heads % config.num_key_value_heads != 0:
            raise ValueError(
                f"Laguna layer {layer_idx} has {heads} query heads and "
                f"{config.num_key_value_heads} KV heads; the query count must be a "
                "multiple of the KV count."
            )
        partial = (
            config.sliding_partial_rotary_factor
            if layer_type == "sliding_attention"
            else config.partial_rotary_factor
        )
        attn_config = dataclasses.replace(
            config, num_attention_heads=heads, partial_rotary_factor=partial
        )
        self.self_attn = LagunaAttention(
            attn_config, config.attn_output_gate, linear_class=linear_class
        )
        if _is_moe_layer(config, layer_idx):
            moe_config = dataclasses.replace(
                config, qmoe_source_paths=(f"model.layers.{layer_idx}.mlp",)
            )
            self.mlp = LagunaSparseMoeBlock(moe_config, linear_class=linear_class)
        else:
            self.mlp = MLP(config, linear_class=linear_class)
        self.input_layernorm = _rms_norm(config)
        self.post_attention_layernorm = _rms_norm(config)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        residual = hidden_states
        attn_out, present_kv = self.self_attn(
            op,
            self.input_layernorm(op, hidden_states),
            attention_bias,
            position_embeddings,
            past_key_value,
        )
        hidden_states = op.Add(residual, op.CastLike(attn_out, residual))
        residual = hidden_states
        hidden_states = self.mlp(op, self.post_attention_layernorm(op, hidden_states))
        return op.Add(residual, op.CastLike(hidden_states, residual)), present_kv


def _rope_module(config: LagunaConfig, *, sliding: bool) -> nn.Module:
    if sliding:
        rope_config = dataclasses.replace(
            config,
            rope_type=config.sliding_rope_type,
            rope_theta=config.sliding_rope_theta,
            partial_rotary_factor=config.sliding_partial_rotary_factor,
            rope_scaling=config.sliding_rope_scaling,
        )
    else:
        rope_config = config
    rotary = initialize_rope(rope_config)
    if rotary is None:
        raise ValueError("Laguna requires rotary position embeddings.")
    return rotary


class LagunaTextModel(nn.Module):
    """Laguna token backbone with one RoPE table and one mask per attention kind."""

    def __init__(self, config: LagunaConfig):
        super().__init__()
        self.config = config
        self._layer_types = _layer_types(config)
        self.embed_tokens = embedding_for_config(config)
        self.layers = nn.ModuleList(
            [
                LagunaDecoderLayer(config, index, kind)
                for index, kind in enumerate(self._layer_types)
            ]
        )
        self.norm = _rms_norm(config)
        kinds = set(self._layer_types)
        self.full_rotary_emb = (
            _rope_module(config, sliding=False) if "full_attention" in kinds else None
        )
        self.sliding_rotary_emb = (
            _rope_module(config, sliding=True) if "sliding_attention" in kinds else None
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        if _float32_residual(self.config):
            hidden_states = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        embeddings: dict[str, tuple] = {}
        if self.full_rotary_emb is not None:
            embeddings["full_attention"] = self.full_rotary_emb(op, position_ids)
        if self.sliding_rotary_emb is not None:
            embeddings["sliding_attention"] = self.sliding_rotary_emb(op, position_ids)

        biases: dict[str, ir.Value | None] = {
            "full_attention": None,
            "sliding_attention": None,
        }
        if attention_mask is not None:
            biases["full_attention"] = create_attention_bias(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                dtype=self.config.dtype,
            )
            if self.sliding_rotary_emb is not None:
                biases["sliding_attention"] = create_attention_bias(
                    op,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sliding_window=self.config.sliding_window,
                    dtype=self.config.dtype,
                )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for kind, layer, past_kv in zip(self._layer_types, self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op, hidden_states, biases[kind], embeddings[kind], past_kv
            )
            present_key_values.append(present_kv)
        return self.norm(op, hidden_states), present_key_values


def _split_fused_experts(key: str, value: torch.Tensor) -> dict[str, torch.Tensor] | None:
    """Split Transformers-5 float ``experts.{gate_up,down}_proj`` banks per expert."""
    if key.endswith(".mlp.experts.gate_up_proj") and value.ndim == 3:
        prefix = key[: -len("gate_up_proj")]
        half = value.shape[1] // 2
        result = {}
        for expert in range(value.shape[0]):
            result[f"{prefix}{expert}.gate_proj.weight"] = value[expert, :half]
            result[f"{prefix}{expert}.up_proj.weight"] = value[expert, half:]
        return result
    if key.endswith(".mlp.experts.down_proj") and value.ndim == 3:
        prefix = key[: -len("down_proj")]
        return {
            f"{prefix}{expert}.down_proj.weight": value[expert]
            for expert in range(value.shape[0])
        }
    return None


class LagunaCausalLMModel(CausalLMModel):
    """Causal LM for Poolside Laguna (XS 2.1, S 2.1, M.1).

    HuggingFace model type: ``laguna``. Routed experts export as one
    ``com.microsoft.QMoE`` node per layer for Olive INT4 checkpoints whose
    experts are stored as fused ``experts.{gate_up,down}_proj`` sidecars, and
    as a loop over float expert MLPs otherwise.
    """

    category: str = "Mixture of Experts"
    config_class = LagunaConfig

    def __init__(self, config: ArchitectureConfig):
        if not isinstance(config, LagunaConfig):
            raise TypeError("LagunaCausalLMModel requires LagunaConfig.")
        nn.Module.__init__(self)
        self.config = config
        self.model = LagunaTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if effective_tie_word_embeddings(config) and isinstance(
            self.model.embed_tokens, Embedding
        ):
            self.lm_head.weight = self.model.embed_tokens.weight

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map hub or Transformers-5 Laguna weights onto the exported decoder.

        The router bias moves from ``mlp.experts.`` to ``mlp.gate.`` and
        Transformers' ``shared_experts`` maps to the hub's ``shared_expert``. Float
        fused expert banks are split per expert for the dense fallback; Olive
        INT4 fused sidecars are packed into QMoE ``fc1``/``fc2``.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.endswith(".mlp.experts.e_score_correction_bias"):
                key = key.replace(".mlp.experts.", ".mlp.gate.")
            key = key.replace(".mlp.shared_experts.", ".mlp.shared_expert.")
            split = _split_fused_experts(key, value)
            if split is None:
                renamed[key] = value
            else:
                renamed.update(split)

        quantization = self.config.quantization_for("model")
        moe_paths = _moe_layer_paths(self.config)
        return preprocess_quantized_weights(
            renamed,
            quantization,
            tie_embeddings=effective_tie_word_embeddings(self.config),
            qmoe_target_path=".mlp" if moe_paths else None,
            qmoe_quant_methods=("gptq", "awq", "olive"),
            qmoe_num_experts=self.config.num_local_experts,
            qmoe_hidden_size=self.config.hidden_size,
            qmoe_intermediate_size=self.config.moe_intermediate_size,
            qmoe_expected_moe_paths=moe_paths,
            qmoe_source_moe_paths=moe_paths,
            defer_non_expert_sidecars=(
                self.config.component_quantization is not None
                or (quantization is not None and quantization.has_module_plan)
            ),
        )
