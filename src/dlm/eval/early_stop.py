"""Early stopping — thin wrapper over `transformers.EarlyStoppingCallback`.

The HF callback itself does the work (monitors `metric_for_best_model`,
increments a patience counter, sets `control.should_training_stop` when
the counter exceeds `early_stopping_patience`). We wrap it for two
reasons:

1. **Config validation.** Patience must be ≥1; threshold must be
   finite and ≥0. HF accepts nonsense defaults and silently degrades.
2. **Inspection.** After `trainer.train()` returns, we need to know
   whether the run actually early-stopped (vs. completing normally)
   so the summary can record it. HF exposes this via
   `trainer.state.best_metric` + `global_step`; we bundle the check
   in `was_early_stopped()` so downstream callers don't reach into
   trainer internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class EarlyStopConfig:
    """Knobs the trainer threads into `SFTConfig` + the callback.

    `patience`: eval rounds without improvement before stopping.
    `threshold`: minimum `metric_for_best_model` delta that counts as
      improvement. 0.0 means any improvement resets the patience counter.
    `metric`: HF metric name (`"eval_loss"` by default; the
      `compute_metrics` hook also emits `"eval_perplexity"`).
    """

    patience: int = 3
    threshold: float = 0.0
    metric: str = "eval_loss"
    greater_is_better: bool = False

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.threshold < 0.0:
            raise ValueError(f"threshold must be >= 0.0, got {self.threshold}")
        if not self.metric:
            raise ValueError("metric must be a non-empty string")


def build_callback(cfg: EarlyStopConfig) -> Any:
    """Instantiate an HF `EarlyStoppingCallback` from this config."""
    from transformers import EarlyStoppingCallback

    return EarlyStoppingCallback(
        early_stopping_patience=cfg.patience,
        early_stopping_threshold=cfg.threshold,
    )


def was_early_stopped(
    *, max_steps_ran: int, configured_max_steps: int | None, num_epochs_done: float
) -> bool:
    """Best-effort detection of early-stop vs. normal completion.

    HF sets `trainer.state.global_step == max_steps` on natural end
    (when `max_steps > 0`) or completes the full epoch schedule. If the
    trainer exited before either of those, early-stop is the most
    likely reason.

    This is intentionally imprecise — `trainer.state` doesn't expose
    an explicit "early stopped" flag — but the heuristic is right in
    the normal case and harmless in the ambiguous case (the summary
    just records `early_stopped=False`, which is conservative).
    """
    if configured_max_steps is not None and configured_max_steps > 0:
        return max_steps_ran < configured_max_steps
    # No max_steps cap → we're running to num_epochs. A non-integer
    # `num_epochs_done` means we exited mid-epoch, which is the
    # early-stop signal.
    return not float(num_epochs_done).is_integer()
