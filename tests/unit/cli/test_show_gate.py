"""`dlm show --json` learned adapter gate surface.

Verifies the `_summarize_gate` reader honors these shapes:
- no on-disk gate config → no `gate` key in the payload
- uniform-mode `gate_config.json` without recorded events → config-only
  entries on `per_adapter` (no mean_weight / sample_count)
- trained-mode with recorded `gate_events` rows → per-adapter statistics
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app
from dlm.doc.parser import parse_file
from dlm.metrics.events import GateEvent, RunStart
from dlm.metrics.recorder import MetricsRecorder
from dlm.store.manifest import Manifest, save_manifest
from dlm.store.paths import for_dlm
from dlm.train.gate.module import GateMetadata
from dlm.train.gate.paths import gate_config_path, gate_dir


def _scaffold(tmp_path: Path) -> Path:
    doc = tmp_path / "doc.dlm"
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(doc), "--base", "smollm2-135m"])
    assert result.exit_code == 0, result.output
    return doc


def _init_store(doc: Path) -> Path:
    parsed = parse_file(doc)
    store = for_dlm(parsed.frontmatter.dlm_id)
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(dlm_id=parsed.frontmatter.dlm_id, base_model="smollm2-135m"),
    )
    return store.root


def _write_gate_config(store_root: Path, meta: GateMetadata) -> None:
    # Mirror the layout used by `dlm.train.gate.paths` without importing a
    # StorePath just to derive the directory.
    gdir = store_root / "adapter" / "_gate"
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "gate_config.json").write_text(json.dumps(meta.to_json()), encoding="utf-8")


def test_no_gate_config_omits_gate_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DLM_HOME", str(tmp_path / "home"))
    doc = _scaffold(tmp_path)
    _init_store(doc)

    runner = CliRunner()
    result = runner.invoke(app, ["show", str(doc), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "gate" not in payload


def test_uniform_mode_surfaces_adapter_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DLM_HOME", str(tmp_path / "home"))
    doc = _scaffold(tmp_path)
    store_root = _init_store(doc)
    _write_gate_config(
        store_root,
        GateMetadata(
            input_dim=576,
            hidden_proj_dim=64,
            adapter_names=("lexer", "runtime"),
            mode="uniform",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["show", str(doc), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    gate = payload["gate"]
    assert gate["mode"] == "uniform"
    assert gate["adapter_names"] == ["lexer", "runtime"]
    assert gate["input_dim"] == 576
    assert gate["hidden_proj_dim"] == 64
    assert gate["last_run_id"] is None
    # No events → fall back to config-only entries (no mean_weight).
    assert gate["per_adapter"] == [
        {"adapter_name": "lexer"},
        {"adapter_name": "runtime"},
    ]


def test_trained_mode_surfaces_per_adapter_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DLM_HOME", str(tmp_path / "home"))
    doc = _scaffold(tmp_path)
    store_root = _init_store(doc)
    _write_gate_config(
        store_root,
        GateMetadata(
            input_dim=576,
            hidden_proj_dim=64,
            adapter_names=("lexer", "runtime"),
            mode="trained",
        ),
    )
    recorder = MetricsRecorder(store_root)
    recorder.record_run_start(RunStart(run_id=1, adapter_version=1, phase="sft", seed=42))
    recorder.record_gate(
        GateEvent(
            run_id=1,
            adapter_name="lexer",
            mean_weight=0.72,
            sample_count=40,
            mode="trained",
        )
    )
    recorder.record_gate(
        GateEvent(
            run_id=1,
            adapter_name="runtime",
            mean_weight=0.28,
            sample_count=18,
            mode="trained",
        )
    )

    runner = CliRunner()
    result = runner.invoke(app, ["show", str(doc), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    gate = payload["gate"]
    assert gate["mode"] == "trained"
    assert gate["last_run_id"] == 1
    per_adapter = {e["adapter_name"]: e for e in gate["per_adapter"]}
    assert per_adapter["lexer"]["mean_weight"] == pytest.approx(0.72)
    assert per_adapter["lexer"]["sample_count"] == 40
    assert per_adapter["lexer"]["mode"] == "trained"
    assert per_adapter["runtime"]["mean_weight"] == pytest.approx(0.28)
    assert per_adapter["runtime"]["sample_count"] == 18


def test_trained_mode_without_events_falls_back_to_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trained gate persisted but no events recorded yet (e.g. recorder
    failed mid-run). The surface still reports the gate using the
    on-disk config's adapter names so `dlm show` flags the gate's
    existence — it just has no statistics yet."""
    monkeypatch.setenv("DLM_HOME", str(tmp_path / "home"))
    doc = _scaffold(tmp_path)
    store_root = _init_store(doc)
    _write_gate_config(
        store_root,
        GateMetadata(
            input_dim=576,
            hidden_proj_dim=64,
            adapter_names=("a", "b", "c"),
            mode="trained",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["show", str(doc), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    gate = payload["gate"]
    assert gate["last_run_id"] is None
    assert gate["per_adapter"] == [
        {"adapter_name": "a"},
        {"adapter_name": "b"},
        {"adapter_name": "c"},
    ]


def test_human_output_renders_gate_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DLM_HOME", str(tmp_path / "home"))
    doc = _scaffold(tmp_path)
    store_root = _init_store(doc)
    _write_gate_config(
        store_root,
        GateMetadata(
            input_dim=576,
            hidden_proj_dim=64,
            adapter_names=("lexer", "runtime"),
            mode="trained",
        ),
    )
    recorder = MetricsRecorder(store_root)
    recorder.record_run_start(RunStart(run_id=1, adapter_version=1, phase="sft", seed=42))
    recorder.record_gate(
        GateEvent(
            run_id=1,
            adapter_name="lexer",
            mean_weight=0.72,
            sample_count=40,
            mode="trained",
        )
    )

    runner = CliRunner()
    result = runner.invoke(app, ["show", str(doc)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "adapter gate" in out
    assert "trained" in out
    assert "lexer" in out
    assert "weight=0.720" in out
    assert "samples=40" in out


def test_gate_dir_helper_matches_expected_layout(tmp_path: Path) -> None:
    """Regression guard: `_write_gate_config` in this test writes to
    `<store>/adapter/_gate/gate_config.json` — which must match the
    path `gate_config_path(store)` produces."""
    from dlm.store.paths import StorePath

    store = StorePath(root=tmp_path / "store")
    store.ensure_layout()
    expected = gate_config_path(store)
    assert expected.parent == gate_dir(store)
    assert expected.parent == store.root / "adapter" / "_gate"
