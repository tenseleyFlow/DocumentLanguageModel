"""Metrics recorder for the learned adapter gate."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dlm.metrics.db import connect
from dlm.metrics.events import GateEvent, RunStart
from dlm.metrics.recorder import MetricsRecorder


@pytest.fixture
def recorder(tmp_path: Path) -> MetricsRecorder:
    rec = MetricsRecorder(store_root=tmp_path)
    # gate_events references runs(run_id) so we need a run row first.
    rec.record_run_start(RunStart(run_id=1, adapter_version=1, phase="sft", seed=42))
    return rec


class TestGateEventSchema:
    def test_populated_at(self) -> None:
        event = GateEvent(
            run_id=1, adapter_name="lexer", mean_weight=0.7, sample_count=32, mode="trained"
        )
        assert event.at != ""
        assert "T" in event.at  # iso-8601 shape

    def test_explicit_at_preserved(self) -> None:
        event = GateEvent(
            run_id=1,
            adapter_name="runtime",
            mean_weight=0.3,
            sample_count=16,
            mode="trained",
            at="2026-04-21T12:00:00+00:00",
        )
        assert event.at == "2026-04-21T12:00:00+00:00"


class TestRecordGate:
    def test_insert_and_read_back(self, recorder: MetricsRecorder, tmp_path: Path) -> None:
        recorder.record_gate(
            GateEvent(
                run_id=1,
                adapter_name="lexer",
                mean_weight=0.7,
                sample_count=32,
                mode="trained",
            )
        )
        recorder.record_gate(
            GateEvent(
                run_id=1,
                adapter_name="runtime",
                mean_weight=0.3,
                sample_count=16,
                mode="trained",
            )
        )
        with connect(tmp_path) as conn:
            rows = list(
                conn.execute(
                    "SELECT adapter_name, mean_weight, sample_count, mode "
                    "FROM gate_events WHERE run_id = 1 ORDER BY adapter_name"
                )
            )
        assert rows == [("lexer", 0.7, 32, "trained"), ("runtime", 0.3, 16, "trained")]

    def test_replace_on_duplicate(self, recorder: MetricsRecorder, tmp_path: Path) -> None:
        """Primary key is (run_id, adapter_name) — inserting twice for the
        same pair overwrites, not duplicates."""
        recorder.record_gate(
            GateEvent(
                run_id=1, adapter_name="lexer", mean_weight=0.5, sample_count=10, mode="trained"
            )
        )
        recorder.record_gate(
            GateEvent(
                run_id=1, adapter_name="lexer", mean_weight=0.9, sample_count=20, mode="trained"
            )
        )
        with connect(tmp_path) as conn:
            rows = list(
                conn.execute("SELECT mean_weight, sample_count FROM gate_events WHERE run_id = 1")
            )
        assert rows == [(0.9, 20)]

    def test_uniform_mode_recorded(self, recorder: MetricsRecorder, tmp_path: Path) -> None:
        recorder.record_gate(
            GateEvent(run_id=1, adapter_name="a", mean_weight=0.5, sample_count=2, mode="uniform")
        )
        with connect(tmp_path) as conn:
            (mode,) = next(
                conn.execute("SELECT mode FROM gate_events WHERE run_id = 1 AND adapter_name='a'")
            )
        assert mode == "uniform"


def test_schema_includes_gate_events_table(tmp_path: Path) -> None:
    """The migration path creates the table unconditionally on connect."""
    with connect(tmp_path) as conn:
        assert isinstance(conn, sqlite3.Connection)
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "gate_events" in tables
