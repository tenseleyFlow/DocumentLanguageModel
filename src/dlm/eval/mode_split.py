"""Post-training val-loss split by row mode (audit-08 N9).

Sprint 19 added `TrainingSummary.val_loss_cpt` / `val_loss_sft` +
`split_loss_by_mode` but left the plumbing to populate them
unwired. This module closes the loop: given a trained `SFTTrainer`
and its `val_ds`, split the dataset by row mode (CPT prose vs SFT
instruction) and run `trainer.evaluate()` on each non-empty subset
to extract per-mode `eval_loss`.

Kept small and pure-wrapper. Heavy eval lives in TRL; we just
group rows and read `eval_loss` out of the returned dict. Unit
tests drive the grouping logic with a mock trainer.
"""

from __future__ import annotations

from typing import Any


def compute_val_loss_by_mode(trainer: Any, val_ds: Any) -> tuple[float | None, float | None]:
    """Return `(val_loss_cpt, val_loss_sft)` from a post-train eval pass.

    Splits `val_ds` into CPT-only and SFT-only subsets using the
    `dlm.train.cpt.runtime.row_mode` classifier, runs
    `trainer.evaluate()` on each non-empty subset, and returns the
    resulting `eval_loss` values. `None` for any mode with no rows
    in the val set.

    Non-fatal: if the eval call raises (stack version drift, NaN
    logits from an undertrained tiny model, etc.) the affected
    mode's loss stays `None`. The summary gets whatever signal is
    reliably extractable without killing the training run.
    """
    from dlm.train.cpt.runtime import row_mode

    if val_ds is None:
        return (None, None)
    try:
        if len(val_ds) == 0:
            return (None, None)
    except TypeError:
        # Not-sized dataset: bail gracefully rather than crashing.
        return (None, None)

    # Group indices by mode — we filter via HF Dataset.select() so we
    # don't duplicate rows into memory.
    cpt_idx: list[int] = []
    sft_idx: list[int] = []
    for i, row in enumerate(val_ds):
        mode = row_mode(row)
        if mode == "cpt":
            cpt_idx.append(i)
        elif mode == "sft":
            sft_idx.append(i)

    cpt_loss = _safe_eval_loss(trainer, val_ds, cpt_idx)
    sft_loss = _safe_eval_loss(trainer, val_ds, sft_idx)
    return (cpt_loss, sft_loss)


def _safe_eval_loss(trainer: Any, val_ds: Any, indices: list[int]) -> float | None:
    """Run `trainer.evaluate(eval_dataset=subset)`; return eval_loss or None."""
    if not indices:
        return None
    try:
        subset = val_ds.select(indices)
    except Exception:
        return None
    try:
        metrics = trainer.evaluate(eval_dataset=subset)
    except Exception:
        return None
    loss = metrics.get("eval_loss") if isinstance(metrics, dict) else None
    if loss is None:
        return None
    try:
        return float(loss)
    except (TypeError, ValueError):
        return None
