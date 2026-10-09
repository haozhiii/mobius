# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mistral 4 (Mistral-Small-4) text decoder.

Mistral 4 is DeepSeek-V3 Multi-head Latent Attention with interleaved YaRN
RoPE, Llama-4 position-dependent query scaling, and a softmax-routed MoE FFN
with one shared expert per layer.

Reference: HuggingFace ``Mistral4ForCausalLM`` and the ``mistral4`` text model
inside ``Mistral3ForConditionalGeneration``.
"""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
from onnxscript import OpBuilder

from mobius._configs import ArchitectureConfig
from mobius.models.deepseek import DeepSeekV3CausalLMModel


def _mistral4_config(config: ArchitectureConfig) -> ArchitectureConfig:
    # HF expresses the MLA rope width as head_dim * partial_rotary_factor
    # (128 * 0.5); DeepSeekV3TextModel already sizes RoPE by qk_rope_head_dim.
    return dataclasses.replace(config, partial_rotary_factor=1.0)


class Mistral4CausalLMModel(DeepSeekV3CausalLMModel):
    """Mistral 4 causal LM: MLA attention with a shared + softmax-routed MoE FFN.

    model_type: mistral4
    """

    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        super().__init__(_mistral4_config(config))


class Mistral4VLDecoderModel(Mistral4CausalLMModel):
    """Mistral 4 text decoder taking ``inputs_embeds`` for the Mistral3 VLM split."""

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values
