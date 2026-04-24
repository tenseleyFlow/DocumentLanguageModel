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
PreferenceMineWriteMode = Literal["staged", "applied", "empty"]


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
class TokenizationEvent:
    """Emitted from `trainer.run` at the end of directive expansion.

    Tracks the tokenized-section cache: how many lookups hit the
    cache, how many missed and required tokenization, and the cache's
    size on disk after the run.
    """

    run_id: int
    total_sections: int
    cache_hits: int
    cache_misses: int
    total_tokenize_seconds: float
    cache_bytes_after: int
    at: str = ""

    def __post_init__(self) -> None:
        if not self.at:
            object.__setattr__(self, "at", _utc_iso())


@dataclass(frozen=True)
class GateEvent:
    """Emitted after the post-SFT gate training pass.

    Records per-adapter routing statistics so operators can see which
    adapter the gate learned to prefer on average. ``mean_weight`` is
    the softmax average across supervising samples; ``sample_count``
    is how many sections trained that adapter. ``mode`` is ``trained``
    when the gate trained normally, ``uniform`` when the cold-start
    fallback fired, or ``diverged`` when gate training produced a
    non-finite loss and the store fell back to gate-less routing —
    ``mean_weight`` and ``sample_count`` are both 0 for diverged rows.
    """

    run_id: int
    adapter_name: str
    mean_weight: float
    sample_count: int
    mode: str  # "trained" | "uniform" | "diverged"
    at: str = ""

    def __post_init__(self) -> None:
        if not self.at:
            object.__setattr__(self, "at", _utc_iso())


@dataclass(frozen=True)
class PreferenceMineEvent:
    """Emitted from `dlm preference mine` after judging completes."""

    run_id: int
    judge_name: str
    sample_count: int
    mined_pairs: int
    skipped_prompts: int
    write_mode: PreferenceMineWriteMode
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
