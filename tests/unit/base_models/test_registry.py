"""Every curated entry is well-formed and passes the audit-02 F04/F21 rules."""

from __future__ import annotations

import re

import pytest

from dlm.base_models import BASE_MODELS, known_keys
from dlm.base_models.registry import _ENTRIES

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class TestLaunchTen:
    def test_minimum_launch_entries(self) -> None:
        # The original launch ten + PaliGemma (Sprint 35 v1). Additional
        # VL bases land in Sprint 35.3; this floor only enforces "don't
        # regress below what we've shipped".
        assert len(BASE_MODELS) >= 11
        assert len(_ENTRIES) >= 11

    def test_entries_keyed_by_key(self) -> None:
        for entry in _ENTRIES:
            assert BASE_MODELS[entry.key] is entry

    def test_known_keys_stable_ordering(self) -> None:
        keys = known_keys()
        assert isinstance(keys, tuple)
        assert list(keys) == list(BASE_MODELS.keys())


class TestRevisionPinning:
    """Audit-02 F11: every registry entry has a real 40-char commit SHA."""

    @pytest.mark.parametrize("entry", _ENTRIES, ids=lambda e: e.key)
    def test_revision_is_40char_hex(self, entry: object) -> None:
        assert _SHA_RE.fullmatch(entry.revision), (  # type: ignore[attr-defined]
            f"{entry.key} has non-conforming revision {entry.revision!r}"  # type: ignore[attr-defined]
        )


class TestLicenseFields:
    """Audit-02 F04 + F21: license + gating fields populated correctly."""

    @pytest.mark.parametrize("entry", _ENTRIES, ids=lambda e: e.key)
    def test_license_spdx_present(self, entry: object) -> None:
        assert entry.license_spdx  # type: ignore[attr-defined]

    @pytest.mark.parametrize("entry", _ENTRIES, ids=lambda e: e.key)
    def test_gated_entries_have_license_url(self, entry: object) -> None:
        if entry.requires_acceptance:  # type: ignore[attr-defined]
            assert entry.license_url, (  # type: ignore[attr-defined]
                f"{entry.key} is gated but has no license_url"  # type: ignore[attr-defined]
            )

    def test_llama_entries_are_gated_and_nonredistributable(self) -> None:
        """Explicit check: shipped Llama entries keep the Meta gating pattern."""
        for key in ("llama-3.2-1b", "llama-3.2-3b", "llama-3.3-8b-instruct"):
            entry = BASE_MODELS[key]
            assert entry.requires_acceptance is True
            assert entry.redistributable is False

    def test_gemma2_entry_is_gated_and_nonredistributable(self) -> None:
        entry = BASE_MODELS["gemma-2-2b-it"]
        assert entry.requires_acceptance is True
        assert entry.redistributable is False

    def test_apache_entries_are_open_and_redistributable(self) -> None:
        for key in (
            "qwen2.5-0.5b",
            "qwen2.5-1.5b",
            "qwen2.5-coder-1.5b",
            "qwen3-1.7b",
            "qwen3-4b",
            "qwen3-8b",
            "smollm3-3b",
            "olmo-2-7b-instruct",
            "smollm2-135m",
            "smollm2-360m",
            "smollm2-1.7b",
        ):
            entry = BASE_MODELS[key]
            assert entry.license_spdx == "Apache-2.0"
            assert entry.requires_acceptance is False
            assert entry.redistributable is True

    def test_phi_is_mit(self) -> None:
        assert BASE_MODELS["phi-3.5-mini"].license_spdx == "MIT"


class TestArchitectureShapes:
    """Templates, gguf_arch, and target_modules match the architecture family."""

    def test_qwen25_entries_use_qwen2_gguf_arch(self) -> None:
        for key, entry in BASE_MODELS.items():
            # Audio-language Qwen2 variants carry a distinct gguf_arch
            # (no llama.cpp converter support) and a multi-modal template.
            if not key.startswith("qwen2.5") or entry.modality != "text":
                continue
            assert entry.gguf_arch == "qwen2"
            assert entry.architecture == "Qwen2ForCausalLM"
            assert entry.template == "chatml"

    def test_llama_entries_use_llama_gguf_arch(self) -> None:
        for key in ("llama-3.2-1b", "llama-3.2-3b", "llama-3.3-8b-instruct"):
            entry = BASE_MODELS[key]
            assert entry.gguf_arch == "llama"
            assert entry.architecture == "LlamaForCausalLM"
            assert entry.template == "llama3"

    def test_phi_uses_fused_qkv_target_modules(self) -> None:
        entry = BASE_MODELS["phi-3.5-mini"]
        assert "qkv_proj" in entry.target_modules
        assert entry.gguf_arch == "phi3"
        assert entry.template == "phi3"

    def test_olmo2_uses_olmo2_converter_path(self) -> None:
        entry = BASE_MODELS["olmo-2-7b-instruct"]
        assert entry.gguf_arch == "olmo2"
        assert entry.architecture == "Olmo2ForCausalLM"
        assert entry.template == "olmo2"

    def test_gemma2_uses_gemma2_converter_path(self) -> None:
        entry = BASE_MODELS["gemma-2-2b-it"]
        assert entry.gguf_arch == "gemma2"
        assert entry.architecture == "Gemma2ForCausalLM"
        assert entry.template == "gemma2"

    def test_size_gb_fp16_monotonic_within_family(self) -> None:
        qwen_sizes = [
            BASE_MODELS[k].size_gb_fp16 for k in ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b")
        ]
        assert qwen_sizes == sorted(qwen_sizes)
        qwen3_sizes = [BASE_MODELS[k].size_gb_fp16 for k in ("qwen3-1.7b", "qwen3-4b", "qwen3-8b")]
        assert qwen3_sizes == sorted(qwen3_sizes)
        smol_sizes = [
            BASE_MODELS[k].size_gb_fp16 for k in ("smollm2-135m", "smollm2-360m", "smollm2-1.7b")
        ]
        assert smol_sizes == sorted(smol_sizes)
