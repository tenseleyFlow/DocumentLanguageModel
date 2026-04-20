"""Metrics integration smoke: train writes rows, queries read them back.

Slow-marked; depends on `trained_store` fixture to actually run a
trainer cycle. Gated on `DLM_ENABLE_SLOW_INTEGRATION=1`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.fixtures.trained_store import TrainedStoreHandle


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("DLM_ENABLE_SLOW_INTEGRATION") != "1",
        reason="set DLM_ENABLE_SLOW_INTEGRATION=1 to opt in",
    ),
]


def test_trained_store_has_metrics_rows(  # pragma: no cover - slow path
    trained_store: TrainedStoreHandle,
) -> None:
    """After the fixture trains, the metrics DB contains run + step rows."""
    from dlm.metrics.queries import recent_runs, steps_for_run

    runs = recent_runs(trained_store.store.root, limit=10)
    assert runs, "trainer.run() did not record any runs"
    latest = runs[0]
    assert latest.status in ("ok", "running"), (
        f"expected 'ok' or 'running', got {latest.status!r}"
    )

    steps = steps_for_run(trained_store.store.root, latest.run_id)
    # The tiny-model fixture runs at least one step.
    assert steps, "no step rows recorded for the latest run"
