"""Typed errors for `dlm.metrics`."""

from __future__ import annotations


class MetricsError(Exception):
    """Base class for metrics-store errors."""


class MetricsSchemaError(MetricsError):
    """Raised when the on-disk SQLite schema doesn't match the expected one.

    Rare — Pydantic-style migrations live outside this module. The metrics
    DB is best-effort; if the schema drifts we prefer to rename and start
    fresh rather than fight it.
    """
