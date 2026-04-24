"""Modality dispatch registry coverage for Sprint 40 substrate aliases."""

from __future__ import annotations

from dlm.base_models import BaseModelSpec
from dlm.modality import TextModality, modality_for


def _minimal_text_spec(*, modality: str = "text") -> BaseModelSpec:
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
            "modality": modality,
        }
    )


def test_text_moe_reuses_text_dispatch() -> None:
    dispatch = modality_for(_minimal_text_spec(modality="text-moe"))
    assert isinstance(dispatch, TextModality)
    assert dispatch.accepts_images is False
    assert dispatch.accepts_audio is False
