"""Reader-side queries against a seeded metrics DB."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from dlm.metrics.events import EvalEvent, RunEnd, RunStart, StepEvent
from dlm.metrics.queries import (
    evals_for_run,
    evals_to_dict,
    latest_run_id,
    recent_runs,
    runs_to_dict,
    steps_for_run,
    steps_to_dict,
)
from dlm.metrics.recorder import MetricsRecorder


def _seed(store_root: Path) -> None:
    """Populate a DB with three runs and a handful of steps/evals."""
    rec = MetricsRecorder(store_root)
    for run_id in (1, 2, 3):
        rec.record_run_start(RunStart(run_id=run_id, adapter_version=run_id, phase="sft", seed=42))
        for step in (10, 20, 30):
            rec.record_step(StepEvent(run_id=run_id, step=step, loss=2.0 - 0.1 * step))
        rec.record_eval(EvalEvent(run_id=run_id, step=30, val_loss=1.5))
        rec.record_run_end(RunEnd(run_id=run_id, status="ok"))


class TestRecentRuns:
    def test_returns_runs_newest_first(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        runs = recent_runs(tmp_path, limit=10)
        assert [r.run_id for r in runs] == [3, 2, 1]

    def test_limit_caps_results(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        runs = recent_runs(tmp_path, limit=2)
        assert len(runs) == 2

    def test_phase_filter(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        runs = recent_runs(tmp_path, phase="sft")
        assert len(runs) == 3
        runs_dpo = recent_runs(tmp_path, phase="dpo")
        assert runs_dpo == []

    def test_run_id_filter(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        runs = recent_runs(tmp_path, run_id=2)
        assert len(runs) == 1
        assert runs[0].run_id == 2

    def test_since_filter_excludes_old_runs(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        # Hack: rewrite one started_at to be far in the past.
        import sqlite3

        conn = sqlite3.connect(str(tmp_path / "metrics.sqlite"))
        old_ts = (datetime.now(UTC) - timedelta(days=30)).isoformat().replace("+00:00", "Z")
        conn.execute("UPDATE runs SET started_at = ? WHERE run_id = 1", (old_ts,))
        conn.commit()
        conn.close()

        # 24h window → run 1 should drop out.
        runs = recent_runs(tmp_path, since=timedelta(hours=24))
        assert [r.run_id for r in runs] == [3, 2]


class TestStepsAndEvals:
    def test_steps_ordered_by_step(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        steps = steps_for_run(tmp_path, run_id=1)
        assert [s.step for s in steps] == [10, 20, 30]

    def test_steps_since_filter(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        steps = steps_for_run(tmp_path, run_id=1, since_step=15)
        assert [s.step for s in steps] == [20, 30]

    def test_evals_for_run(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        evals = evals_for_run(tmp_path, run_id=2)
        assert len(evals) == 1
        assert evals[0].val_loss == 1.5


class TestLatestRunId:
    def test_returns_max(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        assert latest_run_id(tmp_path) == 3

    def test_none_when_empty(self, tmp_path: Path) -> None:
        # Create empty DB
        from dlm.metrics.db import connect

        with connect(tmp_path) as _conn:
            pass
        assert latest_run_id(tmp_path) is None


class TestDictSerialization:
    def test_runs_to_dict_shape(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        runs = recent_runs(tmp_path, limit=1)
        payload = runs_to_dict(runs)
        assert payload[0].keys() == {
            "run_id",
            "started_at",
            "ended_at",
            "adapter_version",
            "phase",
            "seed",
            "status",
        }

    def test_steps_and_evals_to_dict(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        steps = steps_to_dict(steps_for_run(tmp_path, run_id=1))
        assert all({"step", "loss", "lr", "grad_norm", "at"}.issubset(s.keys()) for s in steps)
        evals = evals_to_dict(evals_for_run(tmp_path, run_id=1))
        assert all("val_loss" in e for e in evals)
