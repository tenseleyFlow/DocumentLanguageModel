"""`TrainingSummary` — post-run report serialized to `logs/train-*.summary.json`.

One summary per training run. Captures "how did it go?" in a compact
form that's cheap to read from the CLI (no torch / HF imports needed)
and human-inspectable. The manifest's `training_runs` list links out
to the summary file by path so `dlm show` (Sprint 13) can load details
on demand.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from dlm.io.atomic import write_text


class ProbeOutput(BaseModel):
    """One probe's prompt + generated response + optional reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str
    response: str
    reference: str | None = None
    section_id: str = ""


class TrainingSummary(BaseModel):
    """Canonical post-run report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: int = Field(..., ge=1)
    adapter_version: int = Field(..., ge=1)
    seed: int
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    final_val_perplexity: float | None = None
    retention_loss: float | None = None
    retention_loss_delta: float | None = None
    probes: list[ProbeOutput] = Field(default_factory=list)
    early_stopped: bool = False
    steps: int = Field(0, ge=0)
    duration_seconds: float = Field(0.0, ge=0.0)
    determinism_class: str = "best_effort"


def save_summary(path: Path, summary: TrainingSummary) -> None:
    """Atomically serialize `summary` as pretty JSON.

    Uses the Sprint 04 atomic-write helper so a concurrent CLI reader
    never sees a torn file.
    """
    payload = summary.model_dump(mode="json")
    blob = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    write_text(path, blob)


def load_summary(path: Path) -> TrainingSummary:
    """Inverse of `save_summary`; raises `pydantic.ValidationError` on drift."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    return TrainingSummary.model_validate(data)


def summary_path_for(logs_dir: Path, run_id: int, started_iso: str) -> Path:
    """Match the JSONL log file naming so pairs are easy to eyeball.

    `train-<run_id>-<ts>.summary.json` sits next to the `.jsonl` log
    with the same stem.
    """
    ts = started_iso.replace(":", "").replace("-", "")
    return logs_dir / f"train-{run_id:06d}-{ts}.summary.json"
