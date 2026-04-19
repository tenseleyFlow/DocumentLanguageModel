"""`_previously_accepted` — CLI license gate for export/prompt (audit-04 B4)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dlm.base_models.license import LicenseAcceptance
from dlm.cli.commands import _previously_accepted
from dlm.store.manifest import Manifest, save_manifest


def _manifest_path(tmp_path: Path, *, with_acceptance: bool) -> Path:
    dst = tmp_path / "manifest.json"
    manifest = Manifest(dlm_id="01TEST", base_model="llama-3.2-1b")
    if with_acceptance:
        manifest = manifest.model_copy(
            update={
                "license_acceptance": LicenseAcceptance(
                    accepted_at=datetime.now(UTC).replace(tzinfo=None, microsecond=0),
                    license_url="https://example.com/license",
                    license_spdx="LLAMA-3",
                    via="cli_flag",
                )
            }
        )
    save_manifest(dst, manifest)
    return dst


def test_missing_manifest_returns_false(tmp_path: Path) -> None:
    assert _previously_accepted(tmp_path / "nope.json") is False


def test_manifest_without_acceptance_returns_false(tmp_path: Path) -> None:
    path = _manifest_path(tmp_path, with_acceptance=False)
    assert _previously_accepted(path) is False


def test_manifest_with_acceptance_returns_true(tmp_path: Path) -> None:
    path = _manifest_path(tmp_path, with_acceptance=True)
    assert _previously_accepted(path) is True


def test_malformed_manifest_returns_false(tmp_path: Path) -> None:
    """Corrupt manifest shouldn't let a gated base through — helper returns False
    (then the gated-base resolver refuses)."""
    path = tmp_path / "manifest.json"
    path.write_text("not json", encoding="utf-8")
    assert _previously_accepted(path) is False


@pytest.mark.parametrize("body", ["", "{"])
def test_garbage_manifest_returns_false(tmp_path: Path, body: str) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(body, encoding="utf-8")
    assert _previously_accepted(path) is False
