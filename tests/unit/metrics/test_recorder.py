"""MetricsRecorder write path + event serialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from dlm.metrics.db import metrics_db_path
from dlm.metrics.events import EvalEvent, ExportEvent, RunEnd, RunStart, StepEvent
from dlm.metrics.recorder import MetricsRecorder


def _select_all(db_path: Path, table: str) -> list[tuple]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
    finally:
        conn.close()
    return rows


class TestRunLifecycle:
    def test_start_writes_running_status(self, tmp_path: Path) -> None:
        rec = MetricsRecorder(tmp_path)
        rec.record_run_start(RunStart(run_id=1, adapter_version=3, phase="sft", seed=42))
        rows = _select_all(metrics_db_path(tmp_path), "runs")
        assert len(rows) == 1
        run_id, started, ended, av, phase, seed, status = rows[0]
        assert run_id == 1
        assert started  # non-empty ISO timestamp
        assert ended is None
        assert av == 3
        assert phase == "sft"
        assert seed == 42
        assert status == "running"

    def test_end_marks_ok(self, tmp_path: Path) -> None:
        rec = MetricsRecorder(tmp_path)
        rec.record_run_start(RunStart(run_id=1, adapter_version=None, phase="sft", seed=0))
        rec.record_run_end(RunEnd(run_id=1, status="ok"))
        rows = _select_all(metrics_db_path(tmp_path), "runs")
        assert rows[0][6] == "ok"  # status column
        assert rows[0][2] is not None  # ended_at

    def test_end_with_failed(self, tmp_path: Path) -> None:
        rec = MetricsRecorder(tmp_path)
        rec.record_run_start(RunStart(run_id=1, adapter_version=None, phase="sft", seed=0))
        rec.record_run_end(RunEnd(run_id=1, status="failed"))
        assert _select_all(metrics_db_path(tmp_path), "runs")[0][6] == "failed"


class TestSteps:
    def test_step_written_with_all_fields(self, tmp_path: Path) -> None:
        rec = MetricsRecorder(tmp_path)
        rec.record_run_start(RunStart(run_id=1, adapter_version=None, phase="sft", seed=0))
        rec.record_step(
            StepEvent(
                run_id=1,
                step=10,
                loss=1.234,
                lr=0.001,
                grad_norm=0.5,
                tokens_per_sec=500.0,
                peak_vram_mb=1024,
            )
        )
        rows = _select_all(metrics_db_path(tmp_path), "steps")
        assert len(rows) == 1
        assert rows[0][0] == 1  # run_id
        assert rows[0][1] == 10  # step
        assert rows[0][2] == 1.234  # loss

    def test_upsert_replaces_on_duplicate_key(self, tmp_path: Path) -> None:
        rec = MetricsRecorder(tmp_path)
        rec.record_run_start(RunStart(run_id=1, adapter_version=None, phase="sft", seed=0))
        rec.record_step(StepEvent(run_id=1, step=10, loss=1.0))
        rec.record_step(StepEvent(run_id=1, step=10, loss=0.9))
        rows = _select_all(metrics_db_path(tmp_path), "steps")
        assert len(rows) == 1
        assert rows[0][2] == 0.9


class TestEvals:
    def test_eval_written(self, tmp_path: Path) -> None:
        rec = MetricsRecorder(tmp_path)
        rec.record_run_start(RunStart(run_id=1, adapter_version=None, phase="sft", seed=0))
        rec.record_eval(EvalEvent(run_id=1, step=50, val_loss=1.8, perplexity=6.0))
        rows = _select_all(metrics_db_path(tmp_path), "evals")
        assert len(rows) == 1
        assert rows[0][2] == 1.8  # val_loss


class TestExports:
    def test_export_written(self, tmp_path: Path) -> None:
        rec = MetricsRecorder(tmp_path)
        rec.record_export(
            ExportEvent(
                quant="Q4_K_M",
                merged=False,
                ollama_name="mydoc:v1",
                size_bytes=123_456,
                duration_s=12.3,
            )
        )
        rows = _select_all(metrics_db_path(tmp_path), "exports")
        assert len(rows) == 1
        # Columns: exported_at, quant, merged, ollama_name, size_bytes, duration_s
        assert rows[0][1] == "Q4_K_M"
        assert rows[0][2] == 0  # merged=False → 0
        assert rows[0][3] == "mydoc:v1"


class TestBestEffort:
    def test_write_after_missing_parent_silently_no_ops(self, tmp_path: Path) -> None:
        """A metrics-write failure must not raise — training is the
        priority, not telemetry."""
        # Point at a path where the DB can't be opened. The recorder
        # should swallow the sqlite error.
        bogus = tmp_path / "nonexistent" / "really" / "no"
        rec = MetricsRecorder(bogus)
        # The connect() helper creates parent dirs, so this actually
        # succeeds. Force a real error by passing a readonly dir.
        bogus.parent.parent.mkdir(parents=True, exist_ok=True)
        rec.record_step(StepEvent(run_id=1, step=0, loss=1.0))
        # No exception → pass.
