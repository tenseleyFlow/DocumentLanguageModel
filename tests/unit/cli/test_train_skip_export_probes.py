"""`dlm train --skip-export-probes` threads into resolve_base_model.

Mirrors the flag that `dlm init` has had since Sprint 6-ish. The train
path originally forgot to forward it — a user could `dlm init
--skip-export-probes` a fresh .dlm with a brand-new architecture, then
`dlm train` would re-run the probes and fail. These tests lock the
forwarding in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app


def _write_minimal_dlm(path: Path) -> None:
    path.write_text(
        "---\n"
        "dlm_id: 01TEST0" + "0" * 19 + "\n"
        "base_model: smollm2-135m\n"
        "training:\n"
        "  seed: 42\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )


class _ResolveCaptureError(Exception):
    """Sentinel raised after capturing the kwargs of resolve_base_model.

    Bubbles out of train_cmd and short-circuits the rest of the training
    machinery so the test stays bounded.
    """

    def __init__(self, kwargs: dict[str, Any]) -> None:
        super().__init__("captured")
        self.captured = kwargs


def _patch_resolve(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Swap ``resolve_base_model`` so it records its kwargs then raises."""
    captured: dict[str, Any] = {}

    def fake_resolve(base_model: str, **kwargs: Any) -> Any:
        captured["base_model"] = base_model
        captured.update(kwargs)
        raise _ResolveCaptureError(captured)

    import dlm.base_models

    monkeypatch.setattr(dlm.base_models, "resolve", fake_resolve)
    return captured


class TestTrainSkipExportProbes:
    def test_flag_threads_true(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        captured = _patch_resolve(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(tmp_path), "train", str(doc), "--skip-export-probes"],
        )
        # train_cmd crashed out via our sentinel after resolve_base_model
        # was called; that's the only call site we care about here.
        assert result.exit_code != 0
        assert captured.get("skip_export_probes") is True, (
            f"expected skip_export_probes=True in resolve kwargs; got {captured!r}"
        )

    def test_flag_defaults_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the flag, the call goes out with ``skip_export_probes=False``."""
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        captured = _patch_resolve(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(
            app, ["--home", str(tmp_path), "train", str(doc)]
        )
        assert result.exit_code != 0
        assert captured.get("skip_export_probes") is False, (
            f"expected skip_export_probes=False in resolve kwargs; got {captured!r}"
        )

    def test_flag_appears_in_help(self) -> None:
        """Sanity: typer exposes the flag in ``dlm train --help``."""
        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--skip-export-probes" in result.output
