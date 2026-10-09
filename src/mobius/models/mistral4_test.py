# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig, VisionConfig
from mobius._testing import make_config
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations._weight_loading import apply_weights
from mobius.models.deepseek import DeepSeekMLADecoderLayer
from mobius.models.llava import LLaVAModel
from mobius.models.mistral4 import Mistral4CausalLMModel, Mistral4VLDecoderModel


def _mistral4_config(**overrides):
    values = dict(
        model_type="mistral4",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        partial_rotary_factor=0.5,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        n_group=1,
        topk_group=1,
        n_shared_experts=1,
        scoring_func="softmax",
        norm_topk_prob=True,
        first_k_dense_replace=0,
        vocab_size=64,
    )
    values.update(overrides)
    return make_config(**values)


def test_rope_spans_qk_rope_head_dim_not_half_of_it():
    # HF writes partial_rotary_factor=0.5 relative to head_dim=16; the MLA
    # rope already uses qk_rope_head_dim=8, so halving again would give 4.
    model = Mistral4CausalLMModel(_mistral4_config())
    assert model.config.partial_rotary_factor == pytest.approx(1.0)
    assert isinstance(model.model.layers[0], DeepSeekMLADecoderLayer)


def test_matches_hf_past_original_context_with_cache():
    """Prefill + cached decode beyond original_max_position_embeddings.

    Weights are scaled up from HF's 0.02 init so attention is far from
    uniform; otherwise the 1 + 0.1 * log(1 + floor(pos / 8)) query scale is
    invisible at logit tolerance.
    """
    from transformers import Mistral4Config, Mistral4ForCausalLM

    hf_config = Mistral4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        max_position_embeddings=64,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 4.0,
            "original_max_position_embeddings": 8,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
        },
    )
    torch.manual_seed(0)
    hf_model = Mistral4ForCausalLM(hf_config).eval()
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            if "norm" not in name:
                param.normal_(0.0, 0.15)

    config = ArchitectureConfig.from_transformers(hf_config)
    module = Mistral4CausalLMModel(config)
    pkg = build_from_module(module, config)
    apply_weights(pkg["model"], module.preprocess_weights(dict(hf_model.state_dict())))

    rng = np.random.default_rng(0)
    start, seq = 5, 6  # positions 5..10 straddle original_max_position_embeddings=8
    ids = rng.integers(1, hf_config.vocab_size, size=(1, seq + 1)).astype(np.int64)
    positions = np.arange(start, start + seq + 1, dtype=np.int64)[None]
    with torch.no_grad():
        hf_prefill = hf_model(
            input_ids=torch.from_numpy(ids[:, :seq]),
            position_ids=torch.from_numpy(positions[:, :seq]),
            use_cache=True,
        )
        hf_decode = hf_model(
            input_ids=torch.from_numpy(ids[:, seq:]),
            position_ids=torch.from_numpy(positions[:, seq:]),
            past_key_values=hf_prefill.past_key_values,
            use_cache=True,
        ).logits.numpy()

    session = OnnxModelSession(pkg["model"])
    feeds = {
        "input_ids": ids[:, :seq],
        "attention_mask": np.ones((1, seq), dtype=np.int64),
        "position_ids": positions[:, :seq],
    }
    for value in pkg["model"].graph.inputs:
        if value.name.startswith("past_key_values"):
            shape = [1] + [d if isinstance(d, int) else 0 for d in list(value.shape)[1:]]
            feeds[value.name] = np.zeros(shape, dtype=np.float32)
    prefill = session.run(feeds)
    feeds = {
        "input_ids": ids[:, seq:],
        "attention_mask": np.ones((1, seq + 1), dtype=np.int64),
        "position_ids": positions[:, seq:],
        **{
            name.replace("present.", "past_key_values."): value
            for name, value in prefill.items()
            if name.startswith("present.")
        },
    }
    decode = session.run(feeds)
    session.close()

    np.testing.assert_allclose(prefill["logits"], hf_prefill.logits.numpy(), atol=1e-3)
    np.testing.assert_allclose(decode["logits"], hf_decode, atol=1e-3)


def test_llava_dispatches_mistral4_text_to_mla_moe_decoder():
    config = dataclasses.replace(
        _mistral4_config(),
        image_token_id=63,
        vision=VisionConfig(
            model_type="pixtral",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=16,
            patch_size=4,
        ),
    )
    model = LLaVAModel(config)
    assert isinstance(model.decoder, Mistral4VLDecoderModel)

    experts, mid, hidden = 4, 16, 32
    renamed = model.preprocess_weights(
        {
            "language_model.model.layers.0.mlp.gate.weight": torch.zeros(experts, hidden),
            "language_model.model.layers.0.mlp.experts.gate_up_proj": torch.zeros(
                experts, 2 * mid, hidden
            ),
            "language_model.model.layers.0.mlp.experts.down_proj": torch.zeros(
                experts, hidden, mid
            ),
            "language_model.model.layers.0.self_attn.kv_b_proj.weight": torch.zeros(48, 8),
            "language_model.lm_head.weight": torch.zeros(64, hidden),
        }
    )
    assert "decoder.model.layers.0.mlp.moe.gate.weight" in renamed
    assert "decoder.model.layers.0.mlp.moe.experts.3.up_proj.weight" in renamed
    assert "decoder.model.layers.0.mlp.moe.experts.0.down_proj.weight" in renamed
    assert "decoder.model.layers.0.self_attn.kv_b_proj.weight" in renamed
    assert "decoder.lm_head.weight" in renamed
    assert not any(".mlp.experts." in key for key in renamed)
