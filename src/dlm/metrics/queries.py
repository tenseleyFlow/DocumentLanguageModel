"""Reader-side queries against the metrics DB.

Used by `dlm metrics` and `dlm metrics watch`. All functions take a
`store_root: Path` and open their own read-only connection. SQLite
in WAL mode allows concurrent readers without blocking the trainer's
writer.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dlm.metrics.db import connect


@dataclass(frozen=True)
class RunRow:
    """Shape of one row from `runs`."""

    run_id: int
    started_at: str
    ended_at: str | None
    adapter_version: int | None
    phase: str | None
    seed: int | None
    status: str | None


@dataclass(frozen=True)
class StepRow:
    run_id: int
    step: int
    loss: float | None
    lr: float | None
    grad_norm: float | None
    tokens_per_sec: float | None
    peak_vram_mb: int | None
    at: str


@dataclass(frozen=True)
class EvalRow:
    run_id: int
    step: int
    val_loss: float | None
    perplexity: float | None
    retention: float | None
    at: str


def recent_runs(
    store_root: Path,
    *,
    limit: int = 20,
    phase: str | None = None,
    since: timedelta | None = None,
    run_id: int | None = None,
) -> list[RunRow]:
    """Return the most-recent runs matching the filters.

    Filters compose: `phase="sft"` AND `since=timedelta(hours=24)` AND
    `run_id=4` are all applied. `limit` caps the result set.
    """
    sql = "SELECT run_id, started_at, ended_at, adapter_version, phase, seed, status FROM runs"
    clauses: list[str] = []
    params: list[Any] = []
    if phase is not None:
        clauses.append("phase = ?")
        params.append(phase)
    if run_id is not None:
        clauses.append("run_id = ?")
        params.append(run_id)
    if since is not None:
        cutoff = (datetime.now(UTC) - since).isoformat().replace("+00:00", "Z")
        clauses.append("started_at >= ?")
        params.append(cutoff)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY run_id DESC LIMIT ?"
    params.append(limit)

    with connect(store_root) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [RunRow(*row) for row in rows]


def steps_for_run(store_root: Path, run_id: int, *, since_step: int = 0) -> list[StepRow]:
    """All step rows for `run_id`, ordered by step.

    `since_step` is the exclusive lower bound — `dlm metrics watch`
    uses it to poll only newly-landed rows.
    """
    with connect(store_root) as conn:
        rows = conn.execute(
            "SELECT run_id, step, loss, lr, grad_norm, tokens_per_sec, peak_vram_mb, at "
            "FROM steps WHERE run_id = ? AND step > ? ORDER BY step ASC",
            (run_id, since_step),
        ).fetchall()
    return [StepRow(*row) for row in rows]


def evals_for_run(store_root: Path, run_id: int, *, since_step: int = 0) -> list[EvalRow]:
    """All eval rows for `run_id`, ordered by step."""
    with connect(store_root) as conn:
        rows = conn.execute(
            "SELECT run_id, step, val_loss, perplexity, retention, at "
            "FROM evals WHERE run_id = ? AND step > ? ORDER BY step ASC",
            (run_id, since_step),
        ).fetchall()
    return [EvalRow(*row) for row in rows]


def latest_run_id(store_root: Path) -> int | None:
    """The most-recent `run_id`, or None on empty / missing DB."""
    try:
        with connect(store_root) as conn:
            row = conn.execute("SELECT MAX(run_id) FROM runs").fetchone()
    except sqlite3.Error:
        return None
    if row is None or row[0] is None:
        return None
    return int(row[0])


def runs_to_dict(runs: list[RunRow]) -> list[dict[str, Any]]:
    """JSON-serializable view used by `dlm metrics --json`."""
    return [
        {
            "run_id": r.run_id,
            "started_at": r.started_at,
            "ended_at": r.ended_at,
            "adapter_version": r.adapter_version,
            "phase": r.phase,
            "seed": r.seed,
            "status": r.status,
        }
        for r in runs
    ]


def steps_to_dict(steps: list[StepRow]) -> list[dict[str, Any]]:
    return [
        {
            "step": s.step,
            "loss": s.loss,
            "lr": s.lr,
            "grad_norm": s.grad_norm,
            "tokens_per_sec": s.tokens_per_sec,
            "peak_vram_mb": s.peak_vram_mb,
            "at": s.at,
        }
        for s in steps
    ]


def evals_to_dict(evals: list[EvalRow]) -> list[dict[str, Any]]:
    return [
        {
            "step": e.step,
            "val_loss": e.val_loss,
            "perplexity": e.perplexity,
            "retention": e.retention,
            "at": e.at,
        }
        for e in evals
    ]
