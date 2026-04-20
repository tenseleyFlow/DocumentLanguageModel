"""SQLite schema + WAL + idempotent ensure_schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from dlm.metrics.db import METRICS_DB_FILENAME, connect, ensure_schema, metrics_db_path


class TestMetricsDbPath:
    def test_filename_is_metrics_sqlite(self, tmp_path: Path) -> None:
        assert metrics_db_path(tmp_path).name == METRICS_DB_FILENAME

    def test_under_store_root(self, tmp_path: Path) -> None:
        assert metrics_db_path(tmp_path).parent == tmp_path


class TestConnect:
    def test_creates_schema(self, tmp_path: Path) -> None:
        with connect(tmp_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert tables == {"runs", "steps", "evals", "exports"}

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        with connect(tmp_path) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_idempotent_connect(self, tmp_path: Path) -> None:
        """Re-opening on an existing DB doesn't error or duplicate tables."""
        with connect(tmp_path) as _conn:
            pass
        with connect(tmp_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='runs'"
            ).fetchone()[0]
        assert count == 1


class TestEnsureSchema:
    def test_runs_table_columns(self, tmp_path: Path) -> None:
        with connect(tmp_path) as conn:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(runs)")]
        expected = [
            "run_id",
            "started_at",
            "ended_at",
            "adapter_version",
            "phase",
            "seed",
            "status",
        ]
        assert cols == expected

    def test_steps_composite_primary_key(self, tmp_path: Path) -> None:
        with connect(tmp_path) as conn:
            # Inserting two distinct (run_id, step) rows works;
            # repeating the same pair replaces.
            conn.execute(
                "INSERT INTO runs (run_id, started_at, status) VALUES (1, 'now', 'running')"
            )
            conn.execute(
                "INSERT INTO steps (run_id, step, loss, at) VALUES (1, 1, 0.5, 'now')"
            )
            # Duplicate (1, 1) should violate PK unless we upsert.
            try:
                conn.execute(
                    "INSERT INTO steps (run_id, step, loss, at) VALUES (1, 1, 0.4, 'now')"
                )
                raise AssertionError("duplicate PK accepted")
            except sqlite3.IntegrityError:
                pass

    def test_idempotent_schema(self, tmp_path: Path) -> None:
        """Calling ensure_schema again on the same conn is a noop."""
        db_path = tmp_path / METRICS_DB_FILENAME
        conn = sqlite3.connect(str(db_path))
        ensure_schema(conn)
        ensure_schema(conn)  # should not raise
        conn.close()
