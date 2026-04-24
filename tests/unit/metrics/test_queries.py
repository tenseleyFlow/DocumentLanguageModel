"""Reader-side queries against a seeded metrics DB."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dlm.metrics.events import (
    EvalEvent,
    GateEvent,
    PreferenceMineEvent,
    RunEnd,
    RunStart,
    StepEvent,
    TokenizationEvent,
)
from dlm.metrics.queries import (
    evals_for_run,
    evals_to_dict,
    gate_events_for_run,
    latest_gate_events,
    latest_preference_mining,
    latest_run_id,
    latest_tokenization,
    preference_mining_for_run,
    preference_mining_to_dict,
    preference_mining_totals,
    recent_runs,
    runs_to_dict,
    steps_for_run,
    steps_to_dict,
    tokenization_for_run,
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
    rec.record_tokenization(
        TokenizationEvent(
            run_id=3,
            total_sections=10,
            cache_hits=7,
            cache_misses=3,
            total_tokenize_seconds=0.75,
            cache_bytes_after=4096,
        )
    )
    rec.record_gate(
        GateEvent(
            run_id=2,
            adapter_name="tone",
            mean_weight=0.8,
            sample_count=12,
            mode="trained",
        )
    )
    rec.record_gate(
        GateEvent(
            run_id=2,
            adapter_name="facts",
            mean_weight=0.2,
            sample_count=12,
            mode="trained",
        )
    )
    rec.record_preference_mine(
        PreferenceMineEvent(
            run_id=2,
            judge_name="sway",
            sample_count=4,
            mined_pairs=1,
            skipped_prompts=0,
            write_mode="staged",
        )
    )
    rec.record_preference_mine(
        PreferenceMineEvent(
            run_id=2,
            judge_name="hf:test/reward",
            sample_count=6,
            mined_pairs=2,
            skipped_prompts=3,
            write_mode="applied",
        )
    )


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

    def test_none_on_sqlite_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.metrics.queries as queries_mod

        def _boom(_store_root: Path) -> sqlite3.Connection:
            raise sqlite3.OperationalError("boom")

        monkeypatch.setattr(queries_mod, "connect", _boom)
        assert latest_run_id(tmp_path) is None


class TestTokenizationQueries:
    def test_tokenization_for_run_returns_row_with_hit_rate(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        row = tokenization_for_run(tmp_path, run_id=3)
        assert row is not None
        assert row.cache_hits == 7
        assert row.hit_rate == 0.7

    def test_tokenization_for_run_none_when_table_has_no_row(self, tmp_path: Path) -> None:
        from dlm.metrics.db import connect

        with connect(tmp_path) as _conn:
            pass
        assert tokenization_for_run(tmp_path, run_id=3) is None

    def test_hit_rate_zero_when_total_lookups_is_zero(self) -> None:
        from dlm.metrics.queries import TokenizationRow

        row = TokenizationRow(
            run_id=1,
            total_sections=0,
            cache_hits=0,
            cache_misses=0,
            total_tokenize_seconds=0.0,
            cache_bytes_after=0,
            at="2026-01-01T00:00:00Z",
        )
        assert row.hit_rate == 0.0

    def test_tokenization_for_run_none_on_sqlite_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.metrics.queries as queries_mod

        def _boom(_store_root: Path) -> sqlite3.Connection:
            raise sqlite3.OperationalError("boom")

        monkeypatch.setattr(queries_mod, "connect", _boom)
        assert tokenization_for_run(tmp_path, run_id=1) is None

    def test_latest_tokenization_returns_most_recent_row(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        row = latest_tokenization(tmp_path)
        assert row is not None
        assert row.run_id == 3

    def test_latest_tokenization_none_when_empty(self, tmp_path: Path) -> None:
        from dlm.metrics.db import connect

        with connect(tmp_path) as _conn:
            pass
        assert latest_tokenization(tmp_path) is None

    def test_latest_tokenization_none_on_sqlite_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.metrics.queries as queries_mod

        def _boom(_store_root: Path) -> sqlite3.Connection:
            raise sqlite3.OperationalError("boom")

        monkeypatch.setattr(queries_mod, "connect", _boom)
        assert latest_tokenization(tmp_path) is None


class TestGateQueries:
    def test_gate_events_for_run_returns_rows_sorted_by_adapter(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        rows = gate_events_for_run(tmp_path, run_id=2)
        assert [row.adapter_name for row in rows] == ["facts", "tone"]

    def test_gate_events_for_run_returns_empty_on_sqlite_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.metrics.queries as queries_mod

        def _boom(_store_root: Path) -> sqlite3.Connection:
            raise sqlite3.OperationalError("boom")

        monkeypatch.setattr(queries_mod, "connect", _boom)
        assert gate_events_for_run(tmp_path, run_id=2) == []

    def test_latest_gate_events_returns_latest_run_rows(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        rows = latest_gate_events(tmp_path)
        assert [row.adapter_name for row in rows] == ["facts", "tone"]

    def test_latest_gate_events_empty_when_table_empty(self, tmp_path: Path) -> None:
        from dlm.metrics.db import connect

        with connect(tmp_path) as _conn:
            pass
        assert latest_gate_events(tmp_path) == []

    def test_latest_gate_events_empty_on_sqlite_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.metrics.queries as queries_mod

        def _boom(_store_root: Path) -> sqlite3.Connection:
            raise sqlite3.OperationalError("boom")

        monkeypatch.setattr(queries_mod, "connect", _boom)
        assert latest_gate_events(tmp_path) == []


class TestPreferenceMiningQueries:
    def test_preference_mining_for_run_returns_oldest_first(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        rows = preference_mining_for_run(tmp_path, run_id=2)
        assert [row.judge_name for row in rows] == ["sway", "hf:test/reward"]
        assert [row.write_mode for row in rows] == ["staged", "applied"]

    def test_latest_preference_mining_returns_most_recent_event(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        row = latest_preference_mining(tmp_path)
        assert row is not None
        assert row.judge_name == "hf:test/reward"
        assert row.write_mode == "applied"

    def test_latest_preference_mining_none_when_empty(self, tmp_path: Path) -> None:
        from dlm.metrics.db import connect

        with connect(tmp_path) as _conn:
            pass
        assert latest_preference_mining(tmp_path) is None

    def test_preference_mining_totals_aggregate_across_events(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        totals = preference_mining_totals(tmp_path)
        assert totals is not None
        assert totals.run_count == 1
        assert totals.event_count == 2
        assert totals.total_mined_pairs == 3
        assert totals.total_skipped_prompts == 3

    def test_preference_mining_for_run_returns_empty_on_sqlite_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.metrics.queries as queries_mod

        def _boom(_store_root: Path) -> sqlite3.Connection:
            raise sqlite3.OperationalError("boom")

        monkeypatch.setattr(queries_mod, "connect", _boom)
        assert preference_mining_for_run(tmp_path, run_id=2) == []

    def test_latest_preference_mining_returns_none_on_sqlite_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.metrics.queries as queries_mod

        def _boom(_store_root: Path) -> sqlite3.Connection:
            raise sqlite3.OperationalError("boom")

        monkeypatch.setattr(queries_mod, "connect", _boom)
        assert latest_preference_mining(tmp_path) is None

    def test_preference_mining_totals_none_when_table_empty(self, tmp_path: Path) -> None:
        from dlm.metrics.db import connect

        with connect(tmp_path) as _conn:
            pass
        assert preference_mining_totals(tmp_path) is None

    def test_preference_mining_totals_none_on_sqlite_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.metrics.queries as queries_mod

        def _boom(_store_root: Path) -> sqlite3.Connection:
            raise sqlite3.OperationalError("boom")

        monkeypatch.setattr(queries_mod, "connect", _boom)
        assert preference_mining_totals(tmp_path) is None


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

    def test_preference_mining_to_dict_shape(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        payload = preference_mining_to_dict(preference_mining_for_run(tmp_path, run_id=2))
        assert payload[0].keys() == {
            "event_id",
            "run_id",
            "judge_name",
            "sample_count",
            "mined_pairs",
            "skipped_prompts",
            "write_mode",
            "at",
        }
