"""`PackHeader` + `PackManifest` validation (Sprint 14)."""

from __future__ import annotations

from datetime import datetime

import pytest

from dlm.pack.format import CURRENT_PACK_FORMAT_VERSION, PackHeader, PackManifest


def _valid_header_kwargs() -> dict[str, object]:
    return {
        "pack_format_version": 1,
        "created_at": datetime(2026, 4, 19, 12, 0, 0),
        "tool_version": "0.1.0",
        "content_type": "minimal",
        "platform_hint": "linux",
    }


class TestPackHeader:
    def test_minimal_accepted(self) -> None:
        h = PackHeader.model_validate(_valid_header_kwargs())
        assert h.pack_format_version == 1
        assert h.licensee_acceptance_url is None

    def test_version_must_be_positive(self) -> None:
        kwargs = _valid_header_kwargs() | {"pack_format_version": 0}
        with pytest.raises(ValueError):
            PackHeader.model_validate(kwargs)

    def test_content_type_must_match_literal(self) -> None:
        kwargs = _valid_header_kwargs() | {"content_type": "mystery"}
        with pytest.raises(ValueError):
            PackHeader.model_validate(kwargs)

    def test_extra_forbidden(self) -> None:
        kwargs = _valid_header_kwargs() | {"new_field": "oops"}
        with pytest.raises(ValueError):
            PackHeader.model_validate(kwargs)

    def test_licensee_url_round_trips(self) -> None:
        kwargs = _valid_header_kwargs() | {"licensee_acceptance_url": "https://example.com/accept"}
        h = PackHeader.model_validate(kwargs)
        assert h.licensee_acceptance_url == "https://example.com/accept"

    def test_frozen_forbids_mutation(self) -> None:
        h = PackHeader.model_validate(_valid_header_kwargs())
        with pytest.raises((ValueError, TypeError)):
            h.tool_version = "0.2.0"  # type: ignore[misc]

    def test_current_version_present(self) -> None:
        assert CURRENT_PACK_FORMAT_VERSION >= 1


class TestPackManifest:
    def test_minimal_accepted(self) -> None:
        m = PackManifest(
            dlm_id="01TEST",
            base_model="smollm2-135m",
            adapter_version=0,
            entries={},
            content_sha256="0" * 64,
        )
        assert m.adapter_version == 0

    def test_content_sha256_must_be_64_hex(self) -> None:
        with pytest.raises(ValueError):
            PackManifest(
                dlm_id="x",
                base_model="x",
                adapter_version=0,
                entries={},
                content_sha256="too-short",
            )

    def test_adapter_version_non_negative(self) -> None:
        with pytest.raises(ValueError):
            PackManifest(
                dlm_id="x",
                base_model="x",
                adapter_version=-1,
                entries={},
                content_sha256="0" * 64,
            )
