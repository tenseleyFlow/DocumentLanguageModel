"""Speculative-decoding draft registry (Sprint 12.5)."""

from __future__ import annotations

import pytest

from dlm.base_models import BASE_MODELS
from dlm.base_models.schema import BaseModelSpec
from dlm.export.draft_registry import (
    DRAFT_PAIRS,
    DraftPair,
    resolve_draft,
    validate_registry,
)

_QWEN_3B = BASE_MODELS["qwen2.5-3b"]
_SMOLLM_135M = BASE_MODELS["smollm2-135m"]


# --- Registry validator -----------------------------------------------------


class TestValidateRegistry:
    """Every shipped pair must compile against the real BASE_MODELS."""

    def test_every_pair_references_real_specs(self) -> None:
        validate_registry(BASE_MODELS)  # no raise

    def test_launch_pairs_cover_expected_targets(self) -> None:
        target_keys = {pair.target_key for pair in DRAFT_PAIRS}
        assert "qwen2.5-3b" in target_keys
        assert "llama-3.2-3b" in target_keys
        assert "smollm2-1.7b" in target_keys

    def test_every_pair_shares_template(self) -> None:
        for pair in DRAFT_PAIRS:
            target = BASE_MODELS[pair.target_key]
            draft = BASE_MODELS[pair.draft_registry_key]
            assert target.template == draft.template, pair

    def test_every_pair_shares_tokenizer_pre(self) -> None:
        for pair in DRAFT_PAIRS:
            target = BASE_MODELS[pair.target_key]
            draft = BASE_MODELS[pair.draft_registry_key]
            assert target.tokenizer_pre == draft.tokenizer_pre, pair


class TestValidatorRejectsMismatches:
    """Hand-construct a mini-registry and assert the validator catches drift."""

    def _fake_spec(
        self, *, template: str = "chatml", tokenizer_pre: str = "qwen2"
    ) -> BaseModelSpec:
        return BaseModelSpec.model_validate(
            {
                "key": "fake",
                "hf_id": "org/fake",
                "revision": "0" * 40,
                "architecture": "FakeForCausalLM",
                "params": 1_000_000_000,
                "target_modules": ["q_proj"],
                "template": template,
                "gguf_arch": "fake",
                "tokenizer_pre": tokenizer_pre,
                "license_spdx": "MIT",
                "requires_acceptance": False,
                "redistributable": True,
                "size_gb_fp16": 2.0,
                "context_length": 2048,
                "recommended_seq_len": 1024,
            }
        )

    def test_missing_target_key_raises(self) -> None:
        registry = {"only-draft": self._fake_spec()}
        # Temporarily swap DRAFT_PAIRS by calling validate directly with a
        # synthetic registry that's missing the target. The real
        # DRAFT_PAIRS point at qwen2.5-3b which isn't in this registry.
        with pytest.raises(ValueError, match="target_key 'qwen2.5-3b' not in BASE_MODELS"):
            validate_registry(registry)

    def test_missing_draft_registry_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import dlm.export.draft_registry as mod

        bad_pair = DraftPair(
            target_key="a",
            draft_registry_key="missing",
            upstream_ollama_tag="a:tiny",
            notes="missing draft key",
        )
        monkeypatch.setattr(mod, "DRAFT_PAIRS", (bad_pair,))
        registry = {"a": self._fake_spec()}
        with pytest.raises(ValueError, match="draft_registry_key 'missing' not in BASE_MODELS"):
            validate_registry(registry)

    def test_mismatched_template_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import dlm.export.draft_registry as mod

        bad_pair = DraftPair(
            target_key="a",
            draft_registry_key="b",
            upstream_ollama_tag="a:b",
            notes="template mismatch",
        )
        monkeypatch.setattr(mod, "DRAFT_PAIRS", (bad_pair,))
        registry = {
            "a": self._fake_spec(template="chatml"),
            "b": self._fake_spec(template="llama3"),
        }
        with pytest.raises(ValueError, match="template"):
            validate_registry(registry)

    def test_mismatched_tokenizer_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import dlm.export.draft_registry as mod

        bad_pair = DraftPair(
            target_key="a",
            draft_registry_key="b",
            upstream_ollama_tag="a:b",
            notes="tokenizer mismatch",
        )
        monkeypatch.setattr(mod, "DRAFT_PAIRS", (bad_pair,))
        registry = {
            "a": self._fake_spec(tokenizer_pre="qwen2"),
            "b": self._fake_spec(tokenizer_pre="llama-bpe"),
        }
        with pytest.raises(ValueError, match="tokenizer_pre"):
            validate_registry(registry)


# --- resolve_draft ----------------------------------------------------------


class TestResolveDraft:
    def test_registered_target_returns_tag(self) -> None:
        assert resolve_draft(_QWEN_3B) == "qwen2.5:0.5b"

    def test_unregistered_target_returns_none(self) -> None:
        assert resolve_draft(_SMOLLM_135M) is None

    def test_disabled_returns_none(self) -> None:
        assert resolve_draft(_QWEN_3B, disabled=True) is None

    def test_override_returns_override_verbatim(self) -> None:
        assert resolve_draft(_QWEN_3B, override="custom:1b") == "custom:1b"

    def test_override_wins_even_without_registry_match(self) -> None:
        assert resolve_draft(_SMOLLM_135M, override="x:y") == "x:y"

    def test_disabled_wins_over_override(self) -> None:
        """--no-draft is the nuclear off switch; it beats an explicit --draft."""
        assert resolve_draft(_QWEN_3B, disabled=True, override="should-not-appear") is None
