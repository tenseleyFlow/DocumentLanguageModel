"""MetricsRecorder write path + event serialization."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from dlm.metrics.db import metrics_db_path
from dlm.metrics.events import (
    EvalEvent,
    ExportEvent,
    PreferenceMineEvent,
    RunEnd,
    RunStart,
    StepEvent,
)
from dlm.metrics.recorder import DlmTrainerCallback, MetricsRecorder


def _select_all(db_path: Path, table: str) -> list[tuple]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
    finally:
        conn.close()
    return rows


@contextmanager
def _failing_connect(_store_root: Path) -> Iterator[sqlite3.Connection]:
    raise sqlite3.OperationalError("database is locked")
    yield sqlite3.connect(":memory:")


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


class TestPreferenceMining:
    def test_preference_mine_written_without_run_row(self, tmp_path: Path) -> None:
        rec = MetricsRecorder(tmp_path)
        rec.record_preference_mine(
            PreferenceMineEvent(
                run_id=7,
                judge_name="sway",
                sample_count=4,
                mined_pairs=2,
                skipped_prompts=1,
                write_mode="staged",
            )
        )
        rows = _select_all(metrics_db_path(tmp_path), "preference_mining")
        assert len(rows) == 1
        _, run_id, judge_name, sample_count, mined_pairs, skipped_prompts, write_mode, at = rows[0]
        assert run_id == 7
        assert judge_name == "sway"
        assert sample_count == 4
        assert mined_pairs == 2
        assert skipped_prompts == 1
        assert write_mode == "staged"
        assert at


class TestBestEffort:
    def test_step_write_logs_error_once_per_stream(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Per-step writes stay best-effort, but only one ERROR lands per run."""
        import dlm.metrics.recorder as recorder_mod

        monkeypatch.setattr(recorder_mod, "connect", _failing_connect)
        caplog.set_level(logging.ERROR, logger="dlm.metrics.recorder")
        rec = MetricsRecorder(tmp_path)
        rec.record_step(StepEvent(run_id=1, step=0, loss=1.0))
        rec.record_step(StepEvent(run_id=1, step=1, loss=0.9))

        messages = [record.message for record in caplog.records]
        assert len(messages) == 1
        assert "metrics step write failed" in messages[0]
        assert caplog.records[0].levelno == logging.ERROR

    def test_eval_failure_logs_once_even_after_repeat(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import dlm.metrics.recorder as recorder_mod

        monkeypatch.setattr(recorder_mod, "connect", _failing_connect)
        caplog.set_level(logging.ERROR, logger="dlm.metrics.recorder")
        rec = MetricsRecorder(tmp_path)
        rec.record_eval(EvalEvent(run_id=1, step=1, val_loss=1.0))
        rec.record_eval(EvalEvent(run_id=1, step=2, val_loss=0.9))

        messages = [record.message for record in caplog.records]
        assert len(messages) == 1
        assert "metrics eval write failed" in messages[0]


class TestAnchorWrites:
    def test_run_start_raises_on_sqlite_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.metrics.recorder as recorder_mod

        monkeypatch.setattr(recorder_mod, "connect", _failing_connect)
        rec = MetricsRecorder(tmp_path)

        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            rec.record_run_start(RunStart(run_id=1, adapter_version=1, phase="sft", seed=42))

    def test_run_end_raises_on_sqlite_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.metrics.recorder as recorder_mod

        rec = MetricsRecorder(tmp_path)
        rec.record_run_start(RunStart(run_id=1, adapter_version=1, phase="sft", seed=42))
        monkeypatch.setattr(recorder_mod, "connect", _failing_connect)

        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            rec.record_run_end(RunEnd(run_id=1, status="ok"))


class TestTrainerCallbackCompatibility:
    def test_unknown_hf_lifecycle_hooks_fall_back_to_noop(self, tmp_path: Path) -> None:
        callback = DlmTrainerCallback(MetricsRecorder(tmp_path), run_id=7)

        assert callable(callback.on_train_begin)
        assert callable(callback.on_train_end)
        assert callback.on_train_begin(None, None, None) is None
        assert callback.on_train_end(None, None, None) is None

    def test_non_callback_missing_attr_still_raises(self, tmp_path: Path) -> None:
        callback = DlmTrainerCallback(MetricsRecorder(tmp_path), run_id=7)

        with pytest.raises(AttributeError, match="not_a_callback"):
            _ = callback.not_a_callback
