"""SQLite-backed metrics store (Sprint 26).

One database per `StorePath` at `<store_root>/metrics.sqlite`. WAL mode
is enabled on first connect so a Ctrl-C mid-write leaves a recoverable
DB rather than a corrupt one.

Writes from the trainer happen in the training process; reads happen
from `dlm metrics` (separate process). SQLite WAL supports concurrent
readers + one writer, which matches what we need. No connection pool
— each caller opens, writes, closes.

Schema is idempotent (`CREATE TABLE IF NOT EXISTS`) so calling
`ensure_schema(conn)` on every open is safe and cheap.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


METRICS_DB_FILENAME = "metrics.sqlite"
"""Per-store file. Stored sibling to `manifest.json` + `dlm.lock`."""


_SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id          INTEGER PRIMARY KEY,
        started_at      TEXT NOT NULL,
        ended_at        TEXT,
        adapter_version INTEGER,
        phase           TEXT,
        seed            INTEGER,
        status          TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS steps (
        run_id         INTEGER NOT NULL REFERENCES runs(run_id),
        step           INTEGER NOT NULL,
        loss           REAL,
        lr             REAL,
        grad_norm      REAL,
        tokens_per_sec REAL,
        peak_vram_mb   INTEGER,
        at             TEXT NOT NULL,
        PRIMARY KEY (run_id, step)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evals (
        run_id     INTEGER NOT NULL REFERENCES runs(run_id),
        step       INTEGER NOT NULL,
        val_loss   REAL,
        perplexity REAL,
        retention  REAL,
        at         TEXT NOT NULL,
        PRIMARY KEY (run_id, step)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS exports (
        exported_at TEXT PRIMARY KEY,
        quant       TEXT,
        merged      INTEGER,
        ollama_name TEXT,
        size_bytes  INTEGER,
        duration_s  REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tokenization (
        run_id                  INTEGER PRIMARY KEY REFERENCES runs(run_id),
        total_sections          INTEGER NOT NULL,
        cache_hits              INTEGER NOT NULL,
        cache_misses            INTEGER NOT NULL,
        total_tokenize_seconds  REAL NOT NULL,
        cache_bytes_after       INTEGER NOT NULL,
        at                      TEXT NOT NULL
    )
    """,
]


def metrics_db_path(store_root: Path) -> Path:
    """Location of the per-store metrics DB."""
    return store_root / METRICS_DB_FILENAME


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if absent + enable WAL.

    Called inside every `connect` context manager so the schema is
    guaranteed to exist by the time a caller issues the first query.
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    for statement in _SCHEMA_SQL:
        conn.execute(statement)
    conn.commit()


@contextmanager
def connect(store_root: Path) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection at `store_root/metrics.sqlite` + ensure schema.

    Context-managed: commits on normal exit, rolls back on exception.
    Caller is expected to do their own commits during long transactions;
    the final commit guarantees the most-recent write lands even when
    the caller forgot.
    """
    path = metrics_db_path(store_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    # `isolation_level=None` + explicit BEGIN/COMMIT gives us clear
    # control; we turn on WAL below.
    try:
        ensure_schema(conn)
        yield conn
    finally:
        conn.close()
