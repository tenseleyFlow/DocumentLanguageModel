"""Runtime glue that plugs the CPT utilities into `trainer.run()`.

Two decisions live here:

1. **Schedule selection**: given the user's `training.cpt.schedule`
   setting and the actual row mix, decide whether to patch the
   SFTConfig with DAPT's longer warmup + cosine-with-floor tail.
2. **Override emission**: return the SFTConfig kwargs diff so the
   trainer can apply it without reaching into the scheduler module.

Kept intentionally small so most of the testing weight is on the
pure modules (`schedule.py`, `embed_warmup.py`, `vocab_gap.py`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from dlm.train.cpt.schedule import DAPT_FLOOR_RATIO, DAPT_WARMUP_RATIO

_CPT_FRACTION_THRESHOLD = 0.7
"""When `schedule=auto`, pick DAPT once CPT prose exceeds 70% of rows."""


def row_mode(row: Mapping[str, Any]) -> str:
    """Classify a training row as `"cpt"`, `"sft"`, or `"other"`.

    See `dlm.data.sections_to_rows` for the row-shape contract:

    - ``{"text": ...}`` is PROSE → "cpt"
    - ``{"messages": ...}`` is INSTRUCTION → "sft"
    - ``{"prompt", "chosen", "rejected"}`` is PREFERENCE → "other"
      (preference rows take a separate training phase).
    """
    if "text" in row and row["text"] is not None:
        return "cpt"
    if "messages" in row and row["messages"] is not None:
        return "sft"
    return "other"


def cpt_row_fraction(rows: Iterable[Mapping[str, Any]]) -> float:
    """Return the fraction of `rows` that are CPT-shaped.

    Considers only CPT + SFT rows in the denominator so preference
    triples in a mixed doc don't artificially depress the ratio. An
    empty iterable returns 0.0 (no decision can be made).
    """
    cpt = 0
    sft = 0
    for row in rows:
        mode = row_mode(row)
        if mode == "cpt":
            cpt += 1
        elif mode == "sft":
            sft += 1
    total = cpt + sft
    if total == 0:
        return 0.0
    return cpt / total


def select_schedule(
    schedule_setting: Literal["auto", "dapt", "sft"],
    cpt_fraction: float,
) -> Literal["dapt", "sft"]:
    """Resolve `"auto"` against the row mix; pass through explicit picks.

    Raises `ValueError` on an unknown setting — the pydantic Literal
    already gates most inputs, but runtime users can still smuggle in
    a stale string.
    """
    if schedule_setting == "dapt":
        return "dapt"
    if schedule_setting == "sft":
        return "sft"
    if schedule_setting == "auto":
        return "dapt" if cpt_fraction > _CPT_FRACTION_THRESHOLD else "sft"
    raise ValueError(f"unknown schedule setting: {schedule_setting!r}")


def dapt_sft_config_overrides(
    floor_ratio: float = DAPT_FLOOR_RATIO,
    warmup_ratio: float = DAPT_WARMUP_RATIO,
) -> dict[str, Any]:
    """SFTConfig kwargs that turn HF's cosine into cosine-with-floor.

    Transformers ≥ 4.39 exposes `cosine_with_min_lr` which parks the LR
    at `learning_rate * min_lr_rate` instead of zero — matching our
    DAPT curve. The warmup-ratio bump to 20% is literature-default for
    domain-adaptive pretraining.
    """
    return {
        "lr_scheduler_type": "cosine_with_min_lr",
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_kwargs": {"min_lr_rate": floor_ratio},
    }
