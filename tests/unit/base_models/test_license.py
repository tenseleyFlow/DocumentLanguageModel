"""LicenseAcceptance + require_acceptance (Sprint 12b)."""

from __future__ import annotations

from datetime import datetime

import pytest

from dlm.base_models import BASE_MODELS, GatedModelError, LicenseAcceptance
from dlm.base_models.license import is_gated, require_acceptance


def _non_gated_spec() -> object:
    return BASE_MODELS["smollm2-135m"]


def _gated_spec() -> object:
    return BASE_MODELS["llama-3.2-1b"]


class TestIsGated:
    def test_non_gated_returns_false(self) -> None:
        assert is_gated(_non_gated_spec()) is False  # type: ignore[arg-type]

    def test_gated_returns_true(self) -> None:
        assert is_gated(_gated_spec()) is True  # type: ignore[arg-type]


class TestRequireAcceptance:
    def test_non_gated_returns_none_regardless(self) -> None:
        assert (
            require_acceptance(
                _non_gated_spec(),  # type: ignore[arg-type]
                accept_license=False,
                via="cli_flag",
            )
            is None
        )

    def test_gated_without_accept_raises(self) -> None:
        with pytest.raises(GatedModelError):
            require_acceptance(
                _gated_spec(),  # type: ignore[arg-type]
                accept_license=False,
                via="cli_flag",
            )

    def test_gated_with_accept_returns_record(self) -> None:
        acceptance = require_acceptance(
            _gated_spec(),  # type: ignore[arg-type]
            accept_license=True,
            via="cli_flag",
        )
        assert isinstance(acceptance, LicenseAcceptance)
        assert acceptance.via == "cli_flag"
        spec = _gated_spec()
        assert acceptance.license_spdx == spec.license_spdx  # type: ignore[attr-defined]
        assert acceptance.license_url == spec.license_url  # type: ignore[attr-defined]
        assert isinstance(acceptance.accepted_at, datetime)

    @pytest.mark.parametrize("via", ["cli_flag", "interactive", "frontmatter"])
    def test_via_roundtrip(self, via: str) -> None:
        acc = require_acceptance(
            _gated_spec(),  # type: ignore[arg-type]
            accept_license=True,
            via=via,  # type: ignore[arg-type]
        )
        assert acc is not None
        assert acc.via == via


class TestAcceptanceModel:
    def test_frozen_rejects_mutation(self) -> None:
        acc = require_acceptance(
            _gated_spec(),  # type: ignore[arg-type]
            accept_license=True,
            via="cli_flag",
        )
        assert acc is not None
        with pytest.raises((ValueError, TypeError)):
            acc.via = "interactive"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Unknown keys at deserialize are a programming error."""
        with pytest.raises(ValueError):
            LicenseAcceptance.model_validate(
                {
                    "accepted_at": datetime(2026, 1, 1),
                    "license_url": "https://example.com",
                    "license_spdx": "MIT",
                    "via": "cli_flag",
                    "unknown_extra": "nope",
                }
            )

    def test_via_rejects_unknown(self) -> None:
        with pytest.raises(ValueError):
            LicenseAcceptance.model_validate(
                {
                    "accepted_at": datetime(2026, 1, 1),
                    "license_url": "https://example.com",
                    "license_spdx": "MIT",
                    "via": "magic",
                }
            )
