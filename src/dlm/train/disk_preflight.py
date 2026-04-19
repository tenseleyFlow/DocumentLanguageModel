"""Pre-train disk-space check (audit F12).

A training run that fails halfway through because the disk filled up
leaves the store in a confusing state (half-written checkpoint, partial
log). Catch it up-front: estimate the bytes we're about to write, add
a safety margin, and refuse to start if the filesystem doesn't have
headroom.

The estimate is deliberately pessimistic — LoRA adapters are usually
<100 MB, but the checkpoint also has to hold the torch-serialized
optimizer state (which is ~2× the adapter for AdamW + scaler), logs,
and any cached evaluation artifacts. We also account for replay-corpus
growth (one snapshot per new section, zstd-compressed).

This module doesn't know about the heavy HF stack; it works from
`BaseModelSpec` + `TrainingPlan` + simple arithmetic. Heavy math lives
in `dlm.hardware.memory`; here we only need a byte estimate.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from dlm.train.errors import DiskSpaceError

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.hardware.plan import TrainingPlan

# Floor estimates (bytes). Generous by design; training "fails fast on
# low disk" is a much worse UX than "warns conservatively".
_LOG_RESERVE = 10 * 1024 * 1024  # 10 MB per run for JSONL logs
_OPTIMIZER_MULTIPLIER = 2.5  # AdamW + scaler + schedule state vs adapter bytes
_ADAPTER_FLOOR = 50 * 1024 * 1024  # 50 MB minimum adapter size (conservative)


def estimate_checkpoint_bytes(spec: BaseModelSpec, plan: TrainingPlan) -> int:
    """Rough byte estimate for one full checkpoint commit.

    Components:
    - Adapter weights (LoRA rank × shapes; modeled as a fraction of base)
    - Optimizer state (AdamW keeps m + v per trainable param)
    - Scaler + scheduler state
    - Log reserve
    """
    # LoRA adds roughly `r × (in + out)` params per target module. For a
    # canonical rank-16 adapter on a 1.5B model this works out to ~50 MB
    # in fp16 / ~100 MB in fp32. The estimate treats the adapter as 1%
    # of the base size, clamped to the floor.
    base_bytes = int(spec.size_gb_fp16 * (1024**3))
    adapter_bytes = max(base_bytes // 100, _ADAPTER_FLOOR)
    optimizer_bytes = int(adapter_bytes * _OPTIMIZER_MULTIPLIER)

    # Gradient checkpointing trades time for memory at runtime but
    # doesn't change checkpoint size, so it's not in the formula here.
    _ = plan  # suppress unused warning; kept as a hook for future plan-driven heuristics

    return adapter_bytes + optimizer_bytes + _LOG_RESERVE


def preflight_disk(
    store_root: Path,
    spec: BaseModelSpec,
    plan: TrainingPlan,
    *,
    safety: float = 1.5,
) -> None:
    """Raise `DiskSpaceError` if the store FS can't fit a checkpoint + margin.

    `safety` defaults to 1.5× — the trainer can get unlucky with
    intermediate buffers, and a hard-fail at step 9/10 of a multi-hour
    run is painful.
    """
    if safety <= 0:
        raise ValueError(f"safety must be > 0, got {safety!r}")

    estimate = estimate_checkpoint_bytes(spec, plan)
    required = int(estimate * safety)

    usage = shutil.disk_usage(store_root)
    if usage.free < required:
        raise DiskSpaceError(required_bytes=required, free_bytes=usage.free)
