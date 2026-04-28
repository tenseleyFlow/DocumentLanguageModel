"""Domain dispatcher for `dlm metrics`.

Splits the per-store-metrics query off the CLI plumbing: callers (CLI,
LSP, future web UI) build a `MetricsQuery`, call `gather_metrics`, and
get a typed `MetricsView` with the rows they need to render. The CLI
owns formatting — text, JSON, CSV — and `gather_metrics` doesn't import
typer, rich, or stdout writers.

Imports the queries module dotted (not the individual functions) so
test fixtures that patch `dlm.metrics.queries.<name>` are visible to
this dispatcher at call time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dlm.metrics import queries
from dlm.metrics.queries import EvalRow, PreferenceMineRow, RunRow, StepRow

if TYPE_CHECKING:
    from datetime import timedelta
    from pathlib import Path


@dataclass(frozen=True)
class MetricsQuery:
    """Inputs to `gather_metrics`. All fields validated upstream by the caller."""

    store_root: Path
    run_id: int | None = None
    phase: str | None = None
    since: timedelta | None = None
    limit: int = 20


@dataclass(frozen=True)
class MetricsRunDrilldown:
    """Per-run metrics: the run row plus its steps / evals / preference events."""

    run: RunRow
    steps: tuple[StepRow, ...]
    evals: tuple[EvalRow, ...]
    preference_rows: tuple[PreferenceMineRow, ...]


@dataclass(frozen=True)
class MetricsView:
    """Snapshot of the metrics layer for a given query.

    `drilldown` is set iff `query.run_id is not None` AND a matching run
    exists. `runs` is the row list (always populated for top-level
    listings; on drill-down it contains exactly the matched run when
    found, empty otherwise so the CLI can detect the not-found case).
    """

    runs: tuple[RunRow, ...]
    drilldown: MetricsRunDrilldown | None


def gather_metrics(query: MetricsQuery) -> MetricsView:
    """Read the per-store metrics DB and return a `MetricsView`."""
    runs = tuple(
        queries.recent_runs(
            query.store_root,
            limit=query.limit,
            phase=query.phase,
            since=query.since,
            run_id=query.run_id,
        )
    )

    if query.run_id is None or not runs:
        return MetricsView(runs=runs, drilldown=None)

    run = runs[0]
    steps = tuple(queries.steps_for_run(query.store_root, query.run_id))
    evals = tuple(queries.evals_for_run(query.store_root, query.run_id))
    preference_rows = tuple(queries.preference_mining_for_run(query.store_root, query.run_id))

    return MetricsView(
        runs=runs,
        drilldown=MetricsRunDrilldown(
            run=run,
            steps=steps,
            evals=evals,
            preference_rows=preference_rows,
        ),
    )
