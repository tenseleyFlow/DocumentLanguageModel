"""`dlm train --strict-metrics` plumbing and error surface."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dlm.base_models import BASE_MODELS
from dlm.cli.app import app
from dlm.store.manifest import Manifest, save_manifest
from dlm.store.paths import for_dlm
from tests.fixtures.hardware_mocks import force_mps

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
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


def _normalized_output(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(_ANSI_RE.sub("", text).split())


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

    def fake(
        store: Any,
        parsed: Any,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        captured["store"] = store
        captured["parsed"] = parsed
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(po, "run_phases", fake)
    return captured


@contextmanager
def _failing_connect(_store_root: Path) -> Iterator[sqlite3.Connection]:
    raise sqlite3.OperationalError("database is locked")
    yield sqlite3.connect(":memory:")


class TestTrainStrictMetrics:
    def test_flag_threads_true(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        _seed_store(home)
        _patch_resolve(monkeypatch)
        captured = _patch_run_phases(monkeypatch)

        with force_mps():
            result = CliRunner().invoke(
                app,
                ["--home", str(home), "train", str(doc), "--strict-metrics"],
            )

        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["strict_metrics"] is True

    def test_flag_defaults_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        _seed_store(home)
        _patch_resolve(monkeypatch)
        captured = _patch_run_phases(monkeypatch)

        with force_mps():
            result = CliRunner().invoke(app, ["--home", str(home), "train", str(doc)])

        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["strict_metrics"] is False

    def test_db_failure_surfaces_as_metrics_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.metrics.recorder as recorder_mod

        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        _seed_store(home)
        _patch_resolve(monkeypatch)
        monkeypatch.setattr(recorder_mod, "connect", _failing_connect)

        with force_mps():
            result = CliRunner().invoke(
                app,
                ["--home", str(home), "train", str(doc), "--strict-metrics"],
            )

        normalized = _normalized_output(result).lower()
        assert result.exit_code == 1
        assert "metrics:" in normalized
        assert "database is locked" in normalized

    def test_flag_appears_in_help(self) -> None:
        result = CliRunner().invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--strict-metrics" in _normalized_output(result)
