"""Sprint 40 registry refresh — first 2026 text-family entries."""

from __future__ import annotations

import pytest

from dlm.base_models import BASE_MODELS

_QWEN3_KEYS = ("qwen3-1.7b", "qwen3-4b", "qwen3-8b")


class TestQwen3RegistryEntries:
    @pytest.mark.parametrize("key", _QWEN3_KEYS)
    def test_entry_present(self, key: str) -> None:
        assert key in BASE_MODELS

    @pytest.mark.parametrize("key", _QWEN3_KEYS)
    def test_qwen3_family_uses_expected_architecture(self, key: str) -> None:
        spec = BASE_MODELS[key]
        assert spec.architecture == "Qwen3ForCausalLM"
        assert spec.template == "chatml"
        assert spec.gguf_arch == "qwen3"
        assert spec.tokenizer_pre == "qwen2"

    @pytest.mark.parametrize("key", _QWEN3_KEYS)
    def test_qwen3_family_is_open_and_not_reasoning_tuned(self, key: str) -> None:
        spec = BASE_MODELS[key]
        assert spec.license_spdx == "Apache-2.0"
        assert spec.requires_acceptance is False
        assert spec.redistributable is True
        assert spec.reasoning_tuned is False

    def test_qwen3_8b_uses_upper_bound_size_hint(self) -> None:
        spec = BASE_MODELS["qwen3-8b"]
        assert spec.params == 8_000_000_000
        assert spec.size_gb_fp16 == pytest.approx(16.0)
