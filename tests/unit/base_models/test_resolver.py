"""Registry lookup + `hf:` escape hatch + gating enforcement."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from dlm.base_models import (
    BASE_MODELS,
    GatedModelError,
    UnknownBaseModelError,
    resolve,
)


class TestRegistryLookup:
    def test_known_key_returns_spec(self) -> None:
        spec = resolve("qwen2.5-1.5b")
        assert spec is BASE_MODELS["qwen2.5-1.5b"]

    def test_unknown_key_raises_with_diagnostic(self) -> None:
        with pytest.raises(UnknownBaseModelError) as excinfo:
            resolve("does-not-exist")
        msg = str(excinfo.value)
        assert "does-not-exist" in msg
        # At least one real registry key appears in the preview.
        assert "qwen2.5-1.5b" in msg or "llama-3.2-1b" in msg or "smollm2-135m" in msg

    def test_empty_string_raises(self) -> None:
        with pytest.raises(UnknownBaseModelError):
            resolve("")


class TestGatingEnforcement:
    def test_gated_entry_refused_without_acceptance(self) -> None:
        with pytest.raises(GatedModelError):
            resolve("llama-3.2-1b")

    def test_gated_entry_allowed_with_acceptance(self) -> None:
        spec = resolve("llama-3.2-1b", accept_license=True)
        assert spec.requires_acceptance is True
        assert spec.hf_id == "meta-llama/Llama-3.2-1B-Instruct"

    def test_ungated_entry_ignores_accept_flag(self) -> None:
        # accept_license=True must not affect non-gated bases.
        a = resolve("qwen2.5-1.5b")
        b = resolve("qwen2.5-1.5b", accept_license=True)
        assert a == b


class TestHfEscapeHatch:
    """`hf:` dispatches through `resolve_hf()`; mock out the HF calls."""

    def test_malformed_hf_spec_rejected(self) -> None:
        for malformed in ("hf:no-slash", "hf:/leading-slash", "hf:trailing/"):
            with pytest.raises(UnknownBaseModelError):
                resolve(malformed)

    def test_hf_escape_gates_on_synthesized_spec(self) -> None:
        """If HF reports the repo as gated, we raise GatedModelError."""
        from huggingface_hub.errors import GatedRepoError

        # `HfApi` is imported inside `_synthesize_spec`; patch at source.
        with patch("huggingface_hub.HfApi") as api_cls:
            api_cls.return_value.model_info.side_effect = GatedRepoError("gated", response=Mock())
            with pytest.raises(GatedModelError):
                resolve("hf:meta-llama/made-up")

    def test_hf_escape_unknown_repo_surfaces_as_unknown_base_model(self) -> None:
        from huggingface_hub.errors import RepositoryNotFoundError

        with patch("huggingface_hub.HfApi") as api_cls:
            api_cls.return_value.model_info.side_effect = RepositoryNotFoundError(
                "404", response=Mock()
            )
            with pytest.raises(UnknownBaseModelError):
                resolve("hf:org/deleted")
