# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Export checks for the Step-3.5-Flash decoder.

The hub checkpoint is custom code, and the in-tree ``step3p5`` alias is a
different model. These tests therefore compare the exported graph with a
local copy of the official block math, on a tiny fp32 module.
"""

from __future__ import annotations

import dataclasses
import math
import types

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as functional
from onnxscript import GraphBuilder

from mobius._builder import build_from_module
from mobius._configs import QuantizationConfig, Step3p5Config
from mobius._constants import OPSET_VERSION
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations._weight_loading import apply_weights
from mobius.models.step3p5 import Step3p5CausalLMModel, Step3p5MoE


def _hub_like_config() -> types.SimpleNamespace:
    """Scaled hub fields: 4 decoder layers plus one multi-token-prediction pad."""
    layer_types = [
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    thetas = [5_000_000.0 if kind == "full_attention" else 10_000.0 for kind in layer_types]
    factors = [0.5 if kind == "full_attention" else 1.0 for kind in layer_types]
    return types.SimpleNamespace(
        model_type="step3p5",
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        layer_types=layer_types,
        rope_theta=thetas,
        partial_rotary_factors=factors,
        moe_layers_enum="3,4",
        num_attention_heads=64,
        num_attention_groups=8,
        attention_other_setting={"num_attention_heads": 96, "num_attention_groups": 8},
        head_dim=128,
        hidden_size=4096,
        intermediate_size=11264,
        vocab_size=128896,
        max_position_embeddings=262144,
        sliding_window=512,
        moe_num_experts=288,
        moe_top_k=8,
        moe_intermediate_size=1280,
        share_expert_dim=1280,
        moe_router_scaling_factor=3.0,
        norm_expert_weight=True,
        use_moe_router_bias=True,
        use_head_wise_attn_gate=True,
        yarn_only_types=["full_attention"],
        rope_scaling={
            "rope_type": "llama3",
            "factor": 2.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 32.0,
            "original_max_position_embeddings": 131072,
        },
        swiglu_limits=[0.0, 0.0, 0.0, 7.0, 99.0],
        swiglu_limits_shared=[0.0, 0.0, 0.0, 16.0, 99.0],
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )


def _parity_config() -> Step3p5Config:
    return Step3p5Config(
        hidden_size=16,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        vocab_size=20,
        max_position_embeddings=32,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        rope_type="default",
        rope_theta=10_000.0,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=3,
        sliding_num_attention_heads=4,
        sliding_num_key_value_heads=1,
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        first_k_dense_replace=1,
        full_partial_rotary_factor=0.5,
        sliding_partial_rotary_factor=1.0,
        full_rope_theta=50_000.0,
        sliding_rope_theta=10_000.0,
        yarn_on_full_attention=True,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 2.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 32.0,
            "original_max_position_embeddings": 8,
        },
        original_max_position_embeddings=8,
        routed_scaling_factor=3.0,
        norm_topk_prob=True,
        use_head_wise_attn_gate=True,
        swiglu_limits=[0.0, 0.15],
        swiglu_limits_shared=[0.0, 0.2],
        attn_qk_norm=True,
        dtype=ir.DataType.FLOAT,
        pad_token_id=0,
        disable_qmoe=True,
    )


def test_from_transformers_reads_hub_fields_and_drops_mtp() -> None:
    config = Step3p5Config.from_transformers(_hub_like_config())

    assert config.num_hidden_layers == 4
    assert config.layer_types == [
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
    ]
    assert config.num_attention_heads == 64
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.sliding_num_attention_heads == 96
    assert config.sliding_num_key_value_heads == 8
    assert config.num_local_experts == 288
    assert config.num_experts_per_tok == 8
    assert config.moe_intermediate_size == 1280
    assert config.shared_expert_intermediate_size == 1280
    assert config.routed_scaling_factor == pytest.approx(3.0)
    assert config.norm_topk_prob is True
    assert config.use_expert_bias is True
    assert config.scoring_func == "sigmoid"
    assert config.first_k_dense_replace == 3
    assert config.moe_layer_indices is None
    assert config.full_rope_theta == pytest.approx(5_000_000.0)
    assert config.sliding_rope_theta == pytest.approx(10_000.0)
    assert config.full_partial_rotary_factor == pytest.approx(0.5)
    assert config.sliding_partial_rotary_factor == pytest.approx(1.0)
    assert config.yarn_on_full_attention is True
    assert config.rope_scaling["rope_type"] == "llama3"
    assert config.original_max_position_embeddings == 131072
    assert config.rms_norm_eps == pytest.approx(1e-5)
    assert config.attn_qk_norm is True
    assert config.attn_qkv_bias is False
    assert config.attn_o_bias is False
    assert config.swiglu_limits == [0.0, 0.0, 0.0, 7.0]
    assert config.swiglu_limits_shared == [0.0, 0.0, 0.0, 16.0]
    assert config.use_head_wise_attn_gate is True
    assert config.num_nextn_predict_layers == 1
    assert config.sliding_window == 512


def test_preprocess_splits_expert_banks_and_drops_mtp() -> None:
    module = Step3p5CausalLMModel(_parity_config())
    bank = torch.arange(2 * 8 * 16, dtype=torch.float32).reshape(2, 8, 16)
    routed = module.preprocess_weights(
        {
            "language_model.model.layers.0.moe.gate_proj.weight": bank,
            "language_model.model.embed_tokens.weight": torch.ones(20, 16),
            "model.layers.2.input_layernorm.weight": torch.ones(16),
            "model.layers.1.moe.experts.0.gate_proj.weight": torch.ones(8, 16),
        }
    )

    expert0 = "model.layers.0.moe.experts.0.gate_proj.weight"
    expert1 = "model.layers.0.moe.experts.1.gate_proj.weight"
    assert expert0 in routed
    assert expert1 in routed
    torch.testing.assert_close(routed[expert1], bank[1])
    assert "model.layers.0.moe.gate_proj.weight" not in routed
    assert "model.embed_tokens.weight" in routed
    assert "language_model.model.embed_tokens.weight" not in routed
    assert "model.layers.2.input_layernorm.weight" not in routed
    kept = routed["model.layers.1.moe.experts.0.gate_proj.weight"]
    assert tuple(kept.shape) == (8, 16)


def _numpy_initializer(value: ir.Value) -> np.ndarray:
    array = value.const_value.numpy()
    assert isinstance(array, np.ndarray)
    return array


def _fill_parameters(model: ir.Model) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    weights: dict[str, np.ndarray] = {}
    for name, value in model.graph.initializers.items():
        if value.const_value is not None or value.shape is None:
            continue
        array = rng.normal(0.0, 0.02, size=tuple(int(dim) for dim in value.shape))
        weights[name] = np.ascontiguousarray(array, dtype=np.float32)
    apply_weights(model, {name: torch.from_numpy(array) for name, array in weights.items()})
    return weights


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x @ weight.transpose(-1, -2)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normed = x.float() * torch.rsqrt(variance + eps)
    return normed * (weight.float() + 1.0)


def _apply_rope(
    states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    rotary_dim: int,
) -> torch.Tensor:
    batch, seq, _ = states.shape
    head_dim = states.shape[-1] // num_heads
    shaped = states.reshape(batch, seq, num_heads, head_dim)
    width = head_dim if rotary_dim == 0 else rotary_dim
    rotated = shaped[..., :width]
    passed = shaped[..., width:]
    duplicated_cos = torch.cat((cos, cos), dim=-1).unsqueeze(2)
    duplicated_sin = torch.cat((sin, sin), dim=-1).unsqueeze(2)
    first = rotated[..., : width // 2]
    second = rotated[..., width // 2 :]
    turned = torch.cat((-second, first), dim=-1)
    rotated = rotated * duplicated_cos + turned * duplicated_sin
    return torch.cat((rotated, passed), dim=-1).reshape(batch, seq, num_heads * head_dim)


def _attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    batch, seq, _ = query.shape
    query_heads = query.reshape(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3)
    key_heads = key.reshape(batch, seq, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    value_heads = value.reshape(batch, seq, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    groups = num_heads // num_kv_heads
    if groups != 1:
        key_heads = key_heads.repeat_interleave(groups, dim=1)
        value_heads = value_heads.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query_heads, key_heads.transpose(-1, -2)) * (head_dim**-0.5)
    probs = torch.softmax(scores + bias, dim=-1)
    output = torch.matmul(probs, value_heads)
    return output.permute(0, 2, 1, 3).reshape(batch, seq, num_heads * head_dim)


def _causal_bias(seq: int, window: int | None) -> torch.Tensor:
    query = torch.arange(seq)[:, None]
    key = torch.arange(seq)[None, :]
    allowed = query >= key
    if window is not None:
        allowed = allowed & ((query - key) < window)
    bias = torch.zeros(1, 1, seq, seq)
    return bias.masked_fill(~allowed, torch.finfo(torch.float32).min)


def _swiglu(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    prefix: str,
    limit: float | None,
) -> torch.Tensor:
    gate = functional.silu(_linear(hidden, weights[f"{prefix}.gate_proj.weight"]))
    up = _linear(hidden, weights[f"{prefix}.up_proj.weight"])
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return _linear(gate * up, weights[f"{prefix}.down_proj.weight"])


def _reference_logits(
    config: Step3p5Config,
    weights: dict[str, torch.Tensor],
    cos_full: torch.Tensor,
    sin_full: torch.Tensor,
    cos_sliding: torch.Tensor,
    sin_sliding: torch.Tensor,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    hidden = weights["model.embed_tokens.weight"][input_ids]
    full_bias = _causal_bias(input_ids.shape[1], None)
    sliding_bias = _causal_bias(input_ids.shape[1], config.sliding_window)
    for index, kind in enumerate(config.layer_types or []):
        prefix = f"model.layers.{index}"
        residual = hidden
        normed = _rms_norm(
            hidden, weights[f"{prefix}.input_layernorm.weight"], config.rms_norm_eps
        )
        if kind == "sliding_attention":
            query_heads = config.sliding_num_attention_heads or config.num_attention_heads
            kv_heads = config.sliding_num_key_value_heads or config.num_key_value_heads
            partial = config.sliding_partial_rotary_factor
            cos, sin = cos_sliding, sin_sliding
            bias = sliding_bias
        else:
            query_heads = config.num_attention_heads
            kv_heads = config.num_key_value_heads
            partial = config.full_partial_rotary_factor
            cos, sin = cos_full, sin_full
            bias = full_bias
        rotary_dim = 0 if math.isclose(partial, 1.0) else int(config.head_dim * partial)
        query = _linear(normed, weights[f"{prefix}.self_attn.q_proj.weight"])
        key = _linear(normed, weights[f"{prefix}.self_attn.k_proj.weight"])
        value = _linear(normed, weights[f"{prefix}.self_attn.v_proj.weight"])
        query = _rms_norm(
            query.reshape(*query.shape[:-1], query_heads, config.head_dim),
            weights[f"{prefix}.self_attn.q_norm.weight"],
            config.rms_norm_eps,
        ).reshape(*query.shape[:-1], query_heads * config.head_dim)
        key = _rms_norm(
            key.reshape(*key.shape[:-1], kv_heads, config.head_dim),
            weights[f"{prefix}.self_attn.k_norm.weight"],
            config.rms_norm_eps,
        ).reshape(*key.shape[:-1], kv_heads * config.head_dim)
        query = _apply_rope(
            query, cos[position_ids], sin[position_ids], query_heads, rotary_dim
        )
        key = _apply_rope(key, cos[position_ids], sin[position_ids], kv_heads, rotary_dim)
        attended = _attend(query, key, value, bias, query_heads, kv_heads, config.head_dim)
        gate = torch.sigmoid(_linear(normed, weights[f"{prefix}.self_attn.g_proj.weight"]))
        attended = (
            attended.reshape(*attended.shape[:-1], query_heads, config.head_dim)
            * gate.unsqueeze(-1)
        ).reshape(attended.shape)
        hidden = residual + _linear(attended, weights[f"{prefix}.self_attn.o_proj.weight"])

        residual = hidden
        normed = _rms_norm(
            hidden, weights[f"{prefix}.post_attention_layernorm.weight"], config.rms_norm_eps
        )
        if index >= config.first_k_dense_replace:
            shared_limit = 0.2
            routed_limit = 0.15
            shared = _swiglu(normed, weights, f"{prefix}.share_expert", shared_limit)
            logits = _linear(normed.float(), weights[f"{prefix}.moe.gate.weight"].float())
            probabilities = torch.sigmoid(logits)
            choice = probabilities + weights[f"{prefix}.moe.router_bias"].float()
            _scores, selected = torch.topk(choice, config.num_experts_per_tok or 1, dim=-1)
            gathered = torch.gather(probabilities, -1, selected)
            routing = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20)
            routing = routing * config.routed_scaling_factor
            routed = torch.zeros_like(normed)
            for expert in range(config.num_local_experts or 0):
                expert_out = _swiglu(
                    normed,
                    weights,
                    f"{prefix}.moe.experts.{expert}",
                    routed_limit,
                )
                match = (selected == expert).to(normed.dtype)
                weight = (routing * match).sum(dim=-1, keepdim=True)
                routed = routed + expert_out * weight
            hidden = residual + routed + shared
        else:
            hidden = residual + _swiglu(normed, weights, f"{prefix}.mlp", None)
    hidden = _rms_norm(hidden, weights["model.norm.weight"], config.rms_norm_eps)
    return _linear(hidden, weights["lm_head.weight"])


def test_fp32_prefill_matches_reference_block() -> None:
    config = _parity_config()
    module = Step3p5CausalLMModel(config)
    assert module.kv_cache_specs() == [(2, 8), (1, 8)]
    model = build_from_module(module, config, task="text-generation")["model"]

    names = set(model.graph.initializers)
    assert "model.layers.0.self_attn.g_proj.weight" in names
    assert "model.layers.0.mlp.down_proj.weight" in names
    assert "model.layers.1.share_expert.down_proj.weight" in names
    assert "model.layers.1.moe.experts.0.gate_proj.weight" in names
    assert "model.layers.1.moe.experts.1.up_proj.weight" in names
    q_full = model.graph.initializers["model.layers.0.self_attn.q_proj.weight"]
    q_sliding = model.graph.initializers["model.layers.1.self_attn.q_proj.weight"]
    assert tuple(q_full.shape) == (16, 16)
    assert tuple(q_sliding.shape) == (32, 16)
    bias = model.graph.initializers["model.layers.1.moe.router_bias"]
    assert bias.dtype == ir.DataType.FLOAT

    filled = _fill_parameters(model)
    weights = {name: torch.from_numpy(array) for name, array in filled.items()}
    cos_full = torch.from_numpy(
        _numpy_initializer(model.graph.initializers["model.full_rotary_emb.cos_cache"])
    )
    sin_full = torch.from_numpy(
        _numpy_initializer(model.graph.initializers["model.full_rotary_emb.sin_cache"])
    )
    cos_sliding = torch.from_numpy(
        _numpy_initializer(model.graph.initializers["model.sliding_rotary_emb.cos_cache"])
    )
    sin_sliding = torch.from_numpy(
        _numpy_initializer(model.graph.initializers["model.sliding_rotary_emb.sin_cache"])
    )

    seq = 5
    input_ids = np.array([[1, 3, 5, 7, 2]], dtype=np.int64)
    position_ids = np.arange(seq, dtype=np.int64)[None, :]
    session = OnnxModelSession(model, device="cpu")
    try:
        actual = session.run(
            {
                "input_ids": input_ids,
                "attention_mask": np.ones((1, seq), dtype=np.int64),
                "position_ids": position_ids,
                "past_key_values.0.key": np.zeros((1, 2, 0, 8), dtype=np.float32),
                "past_key_values.0.value": np.zeros((1, 2, 0, 8), dtype=np.float32),
                "past_key_values.1.key": np.zeros((1, 1, 0, 8), dtype=np.float32),
                "past_key_values.1.value": np.zeros((1, 1, 0, 8), dtype=np.float32),
            }
        )
    finally:
        session.close()

    expected = _reference_logits(
        config,
        weights,
        cos_full,
        sin_full,
        cos_sliding,
        sin_sliding,
        torch.from_numpy(input_ids),
        torch.from_numpy(position_ids),
    )
    np.testing.assert_allclose(actual["logits"], expected.numpy(), rtol=1e-4, atol=1e-4)
    assert actual["present.0.key"].shape == (1, 2, seq, 8)
    assert actual["present.1.key"].shape == (1, 1, seq, 8)


_BLOCK = 16


def _int4_config() -> Step3p5Config:
    return dataclasses.replace(
        _parity_config(),
        hidden_size=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        quantization=QuantizationConfig(
            bits=4, group_size=_BLOCK, quant_method="olive", sym=True
        ),
        disable_qmoe=False,
    )


def _pack_int4(codes: np.ndarray) -> np.ndarray:
    """Olive/QMoE nibble order: even column in the low nibble."""
    low = codes[..., 0::2].astype(np.uint8)
    high = codes[..., 1::2].astype(np.uint8)
    return low | (high << 4)


def _dequant_int4(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    blocks = codes.reshape(*codes.shape[:-1], -1, _BLOCK).astype(np.float32) - 8.0
    return (blocks * scales[..., None]).reshape(codes.shape)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def test_int4_build_routes_experts_through_one_qmoe_per_layer() -> None:
    config = _int4_config()
    model = build_from_module(Step3p5CausalLMModel(config), config, task="text-generation")[
        "model"
    ]
    ops = [node.op_type for node in model.graph]
    assert ops.count("QMoE") == 1
    assert "MatMulNBits" in ops
    names = set(model.graph.initializers)
    assert "model.layers.1.moe.fc1_experts_weights" in names
    assert "model.layers.1.moe.fc2_scales" in names
    assert "model.layers.1.moe.experts.0.gate_proj.weight" not in names
    gate = model.graph.initializers["model.layers.1.moe.gate.weight"]
    assert gate.dtype == ir.DataType.FLOAT


def test_int4_qmoe_matches_reference_router_and_experts_on_cpu() -> None:
    """Biased top-k, unbiased weights, scale 3, and the routed SwiGLU clamp.

    QMoE clamps the gate before SiLU and the reference clamps after it. The
    graph must match the QMoE order closely and stay within the analytic
    bound of the reference order.
    """
    config = _int4_config()
    limit = 2.0
    layer = Step3p5MoE(config, routed_limit=limit, linear_class=None)
    assert layer.experts is None

    rng = np.random.default_rng(7)
    experts, hidden, inner = 4, config.hidden_size, 32
    codes = {
        "gate_proj": rng.integers(0, 16, size=(experts, inner, hidden)),
        "up_proj": rng.integers(0, 16, size=(experts, inner, hidden)),
        "down_proj": rng.integers(0, 16, size=(experts, hidden, inner)),
    }
    scales = {
        name: (rng.random((*value.shape[:-1], value.shape[-1] // _BLOCK)) * 0.1 + 0.01).astype(
            np.float32
        )
        for name, value in codes.items()
    }
    checkpoint = {}
    for name, value in codes.items():
        prefix = f"model.layers.1.moe.{name}.weight"
        checkpoint[f"{prefix}_qweight"] = torch.from_numpy(_pack_int4(value))
        checkpoint[f"{prefix}_scales"] = torch.from_numpy(scales[name])
    packed = Step3p5CausalLMModel(config).preprocess_weights(checkpoint)
    for name in ("fc1_experts_weights", "fc1_scales", "fc2_experts_weights", "fc2_scales"):
        tensor = packed[f"model.layers.1.moe.{name}"].numpy()
        assert tuple(getattr(layer, name).shape) == tensor.shape
        getattr(layer, name).const_value = ir.tensor(tensor)

    gate_weight = (rng.standard_normal((experts, hidden)) * 0.3).astype(np.float32)
    router_bias = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    layer.gate.weight.const_value = ir.tensor(gate_weight)
    layer.router_bias.const_value = ir.tensor(router_bias)

    graph = ir.Graph(
        inputs=[],
        outputs=[],
        nodes=[],
        name="step3p5_qmoe",
        opset_imports={"": OPSET_VERSION, "com.microsoft": 1},
    )
    hidden_input = ir.Value(
        name="hidden_states",
        shape=ir.Shape([6, hidden]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    graph.inputs.append(hidden_input)
    output = layer(GraphBuilder(graph).op, hidden_input)
    output.name = "output"
    graph.outputs.append(output)
    assert sum(node.op_type == "QMoE" for node in graph) == 1
    assert sum(node.op_type == "Equal" for node in graph) == 0
    session = ort.InferenceSession(
        ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    tokens = rng.standard_normal((6, hidden)).astype(np.float32)
    (actual,) = session.run(None, {"hidden_states": tokens})

    dense = {name: _dequant_int4(codes[name], scales[name]) for name in codes}
    probabilities = 1.0 / (1.0 + np.exp(-(tokens @ gate_weight.T)))
    selected = np.argsort(-(probabilities + router_bias), axis=-1)[:, :2]
    unbiased = np.argsort(-probabilities, axis=-1)[:, :2]
    assert np.any(np.sort(selected, axis=-1) != np.sort(unbiased, axis=-1))
    routing = np.take_along_axis(probabilities, selected, axis=-1)
    routing = routing / (routing.sum(axis=-1, keepdims=True) + 1e-20) * 3.0

    gate_rows = tokens @ dense["gate_proj"].transpose(0, 2, 1)
    up_rows = np.clip(tokens @ dense["up_proj"].transpose(0, 2, 1), -limit, limit)
    assert np.any(gate_rows > limit)
    qmoe_order = _silu(np.minimum(gate_rows, limit)) * up_rows
    reference_order = np.minimum(_silu(gate_rows), limit) * up_rows

    def combine(activated: np.ndarray) -> np.ndarray:
        expert_out = activated @ dense["down_proj"].transpose(0, 2, 1)
        result = np.zeros_like(tokens)
        for token in range(tokens.shape[0]):
            for slot in range(2):
                expert = selected[token, slot]
                result[token] += routing[token, slot] * expert_out[expert, token]
        return result

    np.testing.assert_allclose(actual, combine(qmoe_order), rtol=1e-3, atol=2e-3)
    down_l1 = np.abs(dense["down_proj"]).sum(axis=-1).max()
    bound = 3.0 * (limit - _silu(np.float32(limit))) * limit * down_l1
    assert np.abs(actual - combine(reference_order)).max() <= bound + 2e-3
