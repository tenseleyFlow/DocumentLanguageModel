"""Reasoning-tuned base-model defaults for Sprint 40."""

from __future__ import annotations

import pytest

from dlm.base_models import BaseModelSpec


def _minimal(*, reasoning_tuned: bool) -> BaseModelSpec:
    return BaseModelSpec.model_validate(
        {
            "key": "demo-1b",
            "hf_id": "org/demo-1b",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "architecture": "DemoForCausalLM",
            "params": 1_000_000_000,
            "target_modules": ["q_proj", "v_proj"],
            "template": "chatml",
            "gguf_arch": "demo",
            "tokenizer_pre": "demo",
            "license_spdx": "Apache-2.0",
            "license_url": None,
            "requires_acceptance": False,
            "redistributable": True,
            "size_gb_fp16": 2.0,
            "context_length": 4096,
            "recommended_seq_len": 2048,
            "reasoning_tuned": reasoning_tuned,
        }
    )


class TestSuggestedPromptTemperature:
    def test_standard_bases_keep_longstanding_default(self) -> None:
        assert _minimal(reasoning_tuned=False).suggested_prompt_temperature == pytest.approx(0.7)

    def test_reasoning_tuned_bases_run_cooler_by_default(self) -> None:
        assert _minimal(reasoning_tuned=True).suggested_prompt_temperature == pytest.approx(0.6)
