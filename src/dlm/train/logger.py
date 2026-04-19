"""Structured per-step JSONL logger for training runs.

One file per run at `logs/train-<run_id>-<started_at>.jsonl`. Every
line is a self-describing JSON object so downstream tools (Sprint 20's
`dlm metrics`, ad-hoc scripts) can parse it without state.

Design
------

- **Atomic line append.** We open in text-append mode and write one
  `json.dumps(...) + "\\n"` per call, then `fh.flush()`. Multi-writer
  isn't a concern (store lock serializes trainers), and a crash
  mid-line is detectable at the JSONL decode layer by downstream
  readers.
- **Banner first, steps after.** The first line is always a
  `"type": "banner"` record with seed, determinism class, pinned
  versions, and the training-plan snapshot. Followed by `"type":
  "step"` records with `step`, `loss`, `lr`, `grad_norm`,
  `tokens_per_sec`, and optional `val_loss` on eval steps.
- **No Rich / tqdm here.** That's a Sprint 13 UX concern. The logger
  is plain JSONL so CI and automation can consume it directly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class Banner:
    """First line of every log file.

    Captures the full run context so a later reader can reconstruct
    what happened without chasing cross-references into the manifest.
    """

    type: str = "banner"
    run_id: int = 0
    seed: int = 0
    determinism_class: str = "best_effort"
    determinism_notes: tuple[str, ...] = ()
    pinned_versions: tuple[tuple[str, str], ...] = ()
    plan: dict[str, Any] | None = None


@dataclass(frozen=True)
class StepRecord:
    """One step's metrics."""

    type: str = "step"
    step: int = 0
    loss: float = 0.0
    lr: float = 0.0
    grad_norm: float | None = None
    tokens_per_sec: float | None = None
    val_loss: float | None = None
    timestamp: str = ""


class StepLogger:
    """File-backed JSONL logger.

    Use as a context manager so the underlying file handle is closed
    on exit even if training raises.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh: TextIO | None = None

    def __enter__(self) -> StepLogger:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *_: object) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @property
    def path(self) -> Path:
        return self._path

    def write_banner(self, banner: Banner) -> None:
        """Write the run banner as the first line."""
        self._write_object(asdict(banner))

    def log_step(
        self,
        *,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float | None = None,
        tokens_per_sec: float | None = None,
        val_loss: float | None = None,
    ) -> None:
        """Append one step record."""
        record = StepRecord(
            step=step,
            loss=loss,
            lr=lr,
            grad_norm=grad_norm,
            tokens_per_sec=tokens_per_sec,
            val_loss=val_loss,
            timestamp=_utc_iso(),
        )
        self._write_object(asdict(record))

    def log_event(self, event_type: str, **fields: Any) -> None:
        """Append an ad-hoc event (e.g., `"type": "oom"`).

        Callers own the shape; we just stamp a timestamp.
        """
        obj: dict[str, Any] = {"type": event_type, "timestamp": _utc_iso()}
        obj.update(_sanitize_fields(fields))
        self._write_object(obj)

    def _write_object(self, obj: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("StepLogger is not open; use `with StepLogger(path):`")
        self._fh.write(json.dumps(obj, sort_keys=True) + "\n")
        self._fh.flush()


def log_path_for(logs_dir: Path, run_id: int) -> Path:
    """Standard log file name: `train-<run_id>-<UTC-ISO>.jsonl`."""
    ts = _utc_iso().replace(":", "").replace("-", "")
    return logs_dir / f"train-{run_id:06d}-{ts}.jsonl"


def _utc_iso() -> str:
    # Tz-naive UTC matches the manifest / replay convention.
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0).isoformat()


def _sanitize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Best-effort conversion of common non-JSON objects."""
    out: dict[str, Any] = {}
    for k, v in fields.items():
        if is_dataclass(v) and not isinstance(v, type):
            out[k] = asdict(v)
        else:
            out[k] = v
    return out
