"""Metrics subsystem — SQLite-backed training/eval/export telemetry.

Heavy optional-dep imports (tensorboard, wandb) stay in the sinks
package; collecting the top-level is cheap.
"""

from __future__ import annotations

from dlm.metrics.db import METRICS_DB_FILENAME, connect, ensure_schema, metrics_db_path
from dlm.metrics.errors import MetricsError, MetricsSchemaError
from dlm.metrics.events import (
    EvalEvent,
    ExportEvent,
    Phase,
    RunEnd,
    RunStart,
    Status,
    StepEvent,
)
from dlm.metrics.recorder import MetricsRecorder

__all__ = [
    "METRICS_DB_FILENAME",
    "EvalEvent",
    "ExportEvent",
    "MetricsError",
    "MetricsRecorder",
    "MetricsSchemaError",
    "Phase",
    "RunEnd",
    "RunStart",
    "Status",
    "StepEvent",
    "connect",
    "ensure_schema",
    "metrics_db_path",
]
