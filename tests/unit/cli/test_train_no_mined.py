"""`dlm train --no-mined` plumbing and help surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dlm.base_models import BASE_MODELS
from dlm.cli.app import app
from dlm.store.manifest import Manifest, save_manifest
from dlm.store.paths import for_dlm
from tests.fixtures.hardware_mocks import force_mps

_DLM_ID = "01TEST0" + "0" * 19


def _write_minimal_dlm(path: Path, *, dlm_id: str = _DLM_ID) -> None:
    path.write_text(
        f"---\ndlm_id: {dlm_id}\nbase_model: smollm2-135m\ntraining:\n  seed: 42\n---\nbody\n",
        encoding="utf-8",
    )


def _seed_store(home: Path, *, dlm_id: str = _DLM_ID) -> None:
    store = for_dlm(dlm_id, home=home)
    store.ensure_layout()
    save_manifest(store.manifest, Manifest(dlm_id=dlm_id, base_model="smollm2-135m"))


def _patch_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    import dlm.base_models

    monkeypatch.setattr(
        dlm.base_models,
        "resolve",
        lambda *args, **kwargs: BASE_MODELS["smollm2-135m"],
    )


def _patch_run_phases(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from dlm.train.preference import phase_orchestrator as po

    captured: dict[str, Any] = {}

    def fake(store: Any, parsed: Any, *args: Any, **kwargs: Any) -> list[Any]:
        captured["store"] = store
        captured["parsed"] = parsed
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(po, "run_phases", fake)
    return captured


class TestTrainNoMined:
    def test_flag_threads_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        _seed_store(home)
        _patch_resolve(monkeypatch)
        captured = _patch_run_phases(monkeypatch)

        with force_mps():
            result = CliRunner().invoke(app, ["--home", str(home), "train", str(doc), "--no-mined"])

        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["include_auto_mined"] is False

    def test_flag_defaults_true(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        _seed_store(home)
        _patch_resolve(monkeypatch)
        captured = _patch_run_phases(monkeypatch)

        with force_mps():
            result = CliRunner().invoke(app, ["--home", str(home), "train", str(doc)])

        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["include_auto_mined"] is True

    def test_flag_appears_in_help(self) -> None:
        result = CliRunner().invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--no-mined" in result.output
