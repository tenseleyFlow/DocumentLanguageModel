"""Typed event dataclasses mirroring the metrics DB rows.

Kept pure-value (no DB coupling) so the recorder layer owns SQL and
callers above (trainer, exporter) just build events and hand them over.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

Phase = Literal["sft", "dpo", "orpo", "cpt"]
Status = Literal["running", "ok", "failed", "cancelled"]


def _utc_iso() -> str:
    """Timestamp we write to SQLite. UTC ISO-8601 with `Z` suffix."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class RunStart:
    """Emitted at the start of `trainer.run`."""

    run_id: int
    adapter_version: int | None
    phase: Phase
    seed: int
    started_at: str = ""

    def __post_init__(self) -> None:
        # Frozen dataclass: set default via object.__setattr__
        if not self.started_at:
            object.__setattr__(self, "started_at", _utc_iso())


@dataclass(frozen=True)
class RunEnd:
    """Emitted at the end of `trainer.run` (success / failure / cancel)."""

    run_id: int
    status: Status
    ended_at: str = ""

    def __post_init__(self) -> None:
        if not self.ended_at:
            object.__setattr__(self, "ended_at", _utc_iso())


@dataclass(frozen=True)
class StepEvent:
    """One training-step log entry."""

    run_id: int
    step: int
    loss: float | None
    lr: float | None = None
    grad_norm: float | None = None
    tokens_per_sec: float | None = None
    peak_vram_mb: int | None = None
    at: str = ""

    def __post_init__(self) -> None:
        if not self.at:
            object.__setattr__(self, "at", _utc_iso())


@dataclass(frozen=True)
class EvalEvent:
    """One eval-step log entry."""

    run_id: int
    step: int
    val_loss: float | None = None
    perplexity: float | None = None
    retention: float | None = None
    at: str = ""

    def __post_init__(self) -> None:
        if not self.at:
            object.__setattr__(self, "at", _utc_iso())


@dataclass(frozen=True)
class ExportEvent:
    """Emitted from `dlm export` on completion."""

    quant: str
    merged: bool
    ollama_name: str | None
    size_bytes: int
    duration_s: float
    exported_at: str = ""

    def __post_init__(self) -> None:
        if not self.exported_at:
            object.__setattr__(self, "exported_at", _utc_iso())
