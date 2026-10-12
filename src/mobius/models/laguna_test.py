# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Export checks for the Poolside Laguna decoder against Transformers' ``laguna``."""

from __future__ import annotations

import types

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import LagunaConfig
from mobius.integrations._weight_loading import apply_weights
from mobius.models.laguna import LagunaCausalLMModel

transformers = pytest.importorskip("transformers")
laguna_hf = pytest.importorskip("transformers.models.laguna")

_LAYER_TYPES = ["full_attention", "sliding_attention", "sliding_attention", "full_attention"]
_ROPE = {
    "full_attention": {
        "rope_type": "yarn",
        "rope_theta": 500_000.0,
        "factor": 4.0,
        "original_max_position_embeddings": 16,
        "beta_slow": 1.0,
        "beta_fast": 32.0,
        "attention_factor": 1.25,
        "partial_rotary_factor": 0.5,
    },
    "sliding_attention": {
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 1.0,
    },
}


def _hf_config(**overrides):
    fields = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        norm_topk_prob=True,
        gating="per-head",
        sliding_window=3,
        layer_types=list(_LAYER_TYPES),
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        num_attention_heads_per_layer=[4, 6, 6, 4],
        moe_routed_scaling_factor=2.5,
        rope_parameters={kind: dict(table) for kind, table in _ROPE.items()},
        tie_word_embeddings=False,
        pad_token_id=0,
    )
    fields.update(overrides)
    return laguna_hf.LagunaConfig(**fields)


def _hf_model(config) -> torch.nn.Module:
    torch.manual_seed(0)
    model = laguna_hf.LagunaForCausalLM(config).to(torch.float32).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("e_score_correction_bias"):
                param.copy_(torch.randn_like(param) * 0.2)
            elif name.endswith("norm.weight"):
                param.copy_(1.0 + torch.randn_like(param) * 0.1)
            else:
                param.copy_(torch.randn_like(param) * 0.1)
    return model


def _to_hub_layout(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite Transformers-5 keys to the hub checkpoint's per-expert names."""
    result = {}
    for key, value in state_dict.items():
        if key.endswith(".mlp.gate.e_score_correction_bias"):
            result[key.replace(".mlp.gate.", ".mlp.experts.")] = value
        elif key.endswith(".mlp.experts.gate_up_proj"):
            prefix = key[: -len("gate_up_proj")]
            half = value.shape[1] // 2
            for expert in range(value.shape[0]):
                result[f"{prefix}{expert}.gate_proj.weight"] = value[expert, :half]
                result[f"{prefix}{expert}.up_proj.weight"] = value[expert, half:]
        elif key.endswith(".mlp.experts.down_proj"):
            prefix = key[: -len("down_proj")]
            for expert in range(value.shape[0]):
                result[f"{prefix}{expert}.down_proj.weight"] = value[expert]
        else:
            result[key.replace(".mlp.shared_experts.", ".mlp.shared_expert.")] = value
    return result


def _run(model: ir.Model, config: LagunaConfig, input_ids, position_ids, past=None):
    batch = input_ids.shape[0]
    feeds = {
        "input_ids": input_ids,
        "attention_mask": np.ones((batch, position_ids[0, -1] + 1), dtype=np.int64),
        "position_ids": position_ids,
    }
    for layer in range(config.num_hidden_layers):
        for kind in ("key", "value"):
            feeds[f"past_key_values.{layer}.{kind}"] = (
                past[f"present.{layer}.{kind}"]
                if past is not None
                else np.zeros(
                    (batch, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
                )
            )
    session = ort.InferenceSession(
        ir.to_proto(model).SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = [output.name for output in session.get_outputs()]
    return dict(zip(names, session.run(names, feeds)))


def _compare_with_hf(hf_model, config: LagunaConfig, model: ir.Model, atol: float) -> None:
    rng = np.random.default_rng(1)
    seq = 7
    input_ids = rng.integers(1, config.vocab_size, size=(1, seq + 1)).astype(np.int64)
    with torch.no_grad():
        expected = hf_model(torch.from_numpy(input_ids)).logits.numpy()

    prefill = _run(model, config, input_ids[:, :seq], np.arange(seq)[None, :])
    np.testing.assert_allclose(prefill["logits"], expected[:, :seq], rtol=1e-4, atol=atol)
    decode = _run(model, config, input_ids[:, seq:], np.array([[seq]]), past=prefill)
    np.testing.assert_allclose(decode["logits"], expected[:, seq:], rtol=1e-4, atol=atol)


def test_from_transformers_reads_laguna_fields() -> None:
    config = LagunaConfig.from_transformers(_hf_config())

    assert config.layer_types == _LAYER_TYPES
    assert config.num_attention_heads_per_layer == [4, 6, 6, 4]
    assert config.dense_layer_indices == [0]
    assert config.attn_output_gate == "per_head"
    assert config.attn_qk_norm and not config.attn_qk_norm_full
    assert config.num_local_experts == 4
    assert config.routed_scaling_factor == pytest.approx(2.5)
    assert config.rope_type == "yarn"
    assert config.partial_rotary_factor == pytest.approx(0.5)
    assert config.rope_scaling["attention_factor"] == pytest.approx(1.25)
    assert config.sliding_rope_type == "default"
    assert config.sliding_rope_theta == pytest.approx(10_000.0)
    assert config.sliding_partial_rotary_factor == pytest.approx(1.0)
    assert config.sliding_rope_scaling is None


def test_from_transformers_reads_hub_remote_code_layout() -> None:
    """Hub remote code keeps a flat rope dict and sparse layers via mlp_only_layers."""
    hub = types.SimpleNamespace(
        model_type="laguna",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        mlp_only_layers=[0],
        decoder_sparse_step=1,
        gating="per-element",
        sliding_window=3,
        layer_types=list(_LAYER_TYPES),
        rope_parameters=dict(_ROPE["full_attention"]),
        swa_rope_parameters=dict(_ROPE["sliding_attention"]),
        partial_rotary_factor=None,
        tie_word_embeddings=False,
    )
    config = LagunaConfig.from_transformers(hub)

    assert config.dense_layer_indices == [0]
    assert config.attn_output_gate == "per_element"
    assert config.num_attention_heads_per_layer is None
    assert config.rope_type == "yarn"
    assert config.partial_rotary_factor == pytest.approx(0.5)
    assert config.sliding_rope_theta == pytest.approx(10_000.0)


@pytest.mark.parametrize("hub_layout", [False, True], ids=["transformers", "hub"])
def test_fp32_prefill_and_decode_match_transformers(hub_layout: bool) -> None:
    hf_config = _hf_config()
    hf_model = _hf_model(hf_config)
    config = LagunaConfig.from_transformers(hf_config)
    config.dtype = ir.DataType.FLOAT
    module = LagunaCausalLMModel(config)
    model = build_from_module(module, config, task="text-generation")["model"]

    names = set(model.graph.initializers)
    assert "model.layers.0.mlp.down_proj.weight" in names
    assert "model.layers.1.mlp.experts.3.up_proj.weight" in names
    assert "model.layers.1.mlp.shared_expert.gate_proj.weight" in names
    assert tuple(model.graph.initializers["model.layers.1.self_attn.g_proj.weight"].shape) == (
        6,
        32,
    )
    assert (
        model.graph.initializers["model.layers.1.mlp.gate.e_score_correction_bias"].dtype
        == ir.DataType.FLOAT
    )

    state_dict = dict(hf_model.state_dict())
    if hub_layout:
        state_dict = _to_hub_layout(state_dict)
    apply_weights(model, module.preprocess_weights(state_dict))
    _compare_with_hf(hf_model, config, model, atol=1e-4)


_BLOCK = 16


def _quantize_rtn(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric INT4 RTN in Olive's layout. Returns (packed, scales, dequantized)."""
    lead, k = weight.shape[:-1], weight.shape[-1]
    blocks = weight.reshape(*lead, k // _BLOCK, _BLOCK)
    scales = blocks.abs().amax(dim=-1).clamp_min(1e-8) / 7.0
    codes = (torch.round(blocks / scales[..., None]).clamp(-8, 7) + 8).to(torch.uint8)
    dequant = ((codes.float() - 8.0) * scales[..., None]).reshape(weight.shape)
    codes = codes.reshape(*lead, k)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), scales.contiguous(), dequant


def _olive_checkpoint(hf_model) -> dict[str, torch.Tensor]:
    """Quantize every linear and fused expert bank; keep router, norms and tables float."""
    checkpoint = {}
    with torch.no_grad():
        for key, value in hf_model.state_dict().items():
            is_bank = key.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj"))
            is_linear = (
                key.endswith(".weight")
                and value.ndim == 2
                and ".layers." in key
                and not key.endswith(".mlp.gate.weight")
            )
            if not (is_bank or is_linear):
                checkpoint[key] = value
                continue
            packed, scales, dequant = _quantize_rtn(value.float())
            checkpoint[f"{key}_qweight"] = packed
            checkpoint[f"{key}_scales"] = scales
            value.copy_(dequant)
    return checkpoint


def test_int4_olive_checkpoint_matches_dequantized_transformers() -> None:
    hf_config = _hf_config()
    hf_model = _hf_model(hf_config)
    checkpoint = _olive_checkpoint(hf_model)
    hf_config.quantization_config = {
        "quant_method": "olive",
        "bits": 4,
        "group_size": _BLOCK,
        "symmetric": True,
        "lm_head": False,
        "embeds": False,
    }
    config = LagunaConfig.from_transformers(hf_config)
    assert config.quantization is not None
    config.dtype = ir.DataType.FLOAT
    module = LagunaCausalLMModel(config)
    model = build_from_module(module, config, task="text-generation")["model"]

    ops = [node.op_type for node in model.graph]
    assert ops.count("QMoE") == 3
    assert "MatMulNBits" in ops
    names = set(model.graph.initializers)
    assert "model.layers.1.mlp.fc1_experts_weights" in names
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" not in names
    initializers = model.graph.initializers
    for name in ("self_attn.g_proj", "mlp.shared_expert.down_proj"):
        assert initializers[f"model.layers.1.{name}.weight"].dtype == ir.DataType.UINT8
    assert initializers["model.layers.1.mlp.gate.weight"].dtype == ir.DataType.FLOAT

    apply_weights(model, module.preprocess_weights(checkpoint))
    _compare_with_hf(hf_model, config, model, atol=2e-4)
