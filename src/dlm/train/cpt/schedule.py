"""LR schedule for domain-adaptation pretraining (DAPT).

The built-in HuggingFace `cosine` scheduler decays linearly through
warmup and then cosines down to zero. For continued pretraining on a
new domain that tends to under-train — we want:

1. A longer warmup (20% vs SFT's 10%) so the embeddings track the new
   token distribution before gradients scale up.
2. A cosine tail that levels off at a non-zero floor (e.g. 10% of
   peak) so late-run updates still contribute instead of vanishing.

Both functions below are pure math — no torch import, no optimizer
handle. `cosine_with_floor_lr` returns the unit-scale multiplier at a
given step; the wiring in `trainer.run()` multiplies by the configured
peak LR. This split keeps the math unit-testable and the
`LambdaLR`/TRL glue trivial.
"""

from __future__ import annotations

import math
from typing import Final

DAPT_WARMUP_RATIO: Final[float] = 0.2
"""Fraction of total steps spent ramping LR from 0 to peak in DAPT mode."""

DAPT_FLOOR_RATIO: Final[float] = 0.1
"""Cosine decay asymptote for DAPT: end LR = peak × 0.1 instead of 0."""


def cosine_with_floor_lr(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    floor_ratio: float = DAPT_FLOOR_RATIO,
) -> float:
    """Unit-scale LR multiplier (0.0 .. 1.0) for a cosine-with-floor schedule.

    - ``step < warmup_steps``: linear ramp 0 → 1.
    - ``warmup_steps <= step < total_steps``: cosine decay from 1
      down to ``floor_ratio`` (never below).
    - ``step >= total_steps``: clamp at ``floor_ratio``.

    Parameters are validated because silently returning a nonsense
    curve under bad inputs is the worst outcome for training.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    if warmup_steps >= total_steps:
        raise ValueError(f"warmup_steps ({warmup_steps}) must be < total_steps ({total_steps})")
    if not 0.0 <= floor_ratio <= 1.0:
        raise ValueError(f"floor_ratio must be in [0.0, 1.0], got {floor_ratio}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    if step < warmup_steps:
        return step / warmup_steps

    if step >= total_steps:
        return floor_ratio

    decay_progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return floor_ratio + (1.0 - floor_ratio) * cosine


def dapt_warmup_steps(total_steps: int, ratio: float = DAPT_WARMUP_RATIO) -> int:
    """Round `total_steps * ratio` to an int ≥1 (unless `total_steps==0`)."""
    if total_steps <= 0:
        return 0
    return max(1, int(round(total_steps * ratio)))
