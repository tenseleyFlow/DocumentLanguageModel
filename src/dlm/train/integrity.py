"""Post-training weight-integrity gate.

Invariant
---------

Training must never persist an adapter whose weights contain NaN or
infinite values. Downstream consumers (inference, export, evaluation)
would then silently produce NaN logits — and the `dlm train` exit code
would be 0, giving the user no signal that the run failed.

This module walks the trainable parameters of a wrapped PEFT model and
asserts every element is finite. The check runs inside the
`_write_checkpoint` writer, between `save_model` and the caller's
atomic pointer flip. If the check fails, the exception propagates up
through `commit_version` — which by contract leaves the pending
version directory on disk and does **not** flip `current.txt`. The
trainer then renames the pending dir to a `-rejected` suffix so the
next `allocate_next_version` skips it (the suffix makes the directory
name unparseable as `vNNNN`) and the user can inspect the bad weights
for postmortem.

Only trainable parameters are checked. Frozen base-model weights are
both huge and (by construction) unchanged by training, so validating
them is wasted I/O without corresponding signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dlm.train.errors import TrainingError

if TYPE_CHECKING:
    pass


class NaNEvalError(TrainingError):
    """The trainer's final eval_loss was NaN or infinite — refusing to commit.

    Fires in the trainer orchestrator after `summarize_eval_state` and
    before the two-phase commit. Redundant with `NaNWeightsError`: when
    eval diverges, the weights are usually already corrupt, so both
    gates catch the same failure from different angles. The redundancy
    is intentional — catching it twice is cheap, catching it zero times
    (the original bug) silently poisons downstream consumers.
    """

    def __init__(self, value: float) -> None:
        super().__init__(
            f"final eval_loss is non-finite ({value!r}) — refusing to commit "
            "adapter. Lower LR, add warmup, or check dataset for NaN-producing inputs."
        )
        self.value = value


def assert_eval_finite(log_history: list[Any]) -> None:
    """Raise `NaNEvalError` if the last `eval_loss` entry is non-finite.

    No-op when `log_history` is empty or contains no eval entries —
    eval is optional, and runs without a `eval_dataset` legitimately
    have no eval_loss to check.
    """
    import math

    for entry in reversed(log_history):
        if not isinstance(entry, dict):
            continue
        value = entry.get("eval_loss")
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise NaNEvalError(float(value))
            return  # last eval was finite — good
    # No eval entries — the caller's contract is "check iff eval ran".


class NaNWeightsError(TrainingError):
    """A trained adapter contains non-finite weights — refusing to persist.

    `offending` is the list of parameter names that failed the check.
    Truncated to the first 20 names for readability; the full list is
    available via the `full_offending` attribute.
    """

    def __init__(self, offending: list[str]) -> None:
        display = offending[:20]
        suffix = f" (and {len(offending) - 20} more)" if len(offending) > 20 else ""
        super().__init__(
            f"adapter weights contain NaN/inf in {len(offending)} tensor(s) — "
            f"refusing to persist. Offenders: {', '.join(display)}{suffix}"
        )
        self.offending = display
        self.full_offending = offending


@dataclass(frozen=True)
class FiniteCheckResult:
    """Outcome of a finite-weight audit, used by unit tests + the gate."""

    checked: int
    offending: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.offending


def audit_trainable_finite(model: Any) -> FiniteCheckResult:
    """Walk `model.named_parameters()` and flag non-finite trainable tensors.

    Only `requires_grad=True` parameters are inspected — these are the
    LoRA `lora_A` / `lora_B` / `modules_to_save` tensors that training
    actually updates. Frozen base weights are skipped.

    Returns a `FiniteCheckResult` even when everything is finite, so
    callers can log the count of checked tensors.
    """
    import torch

    offending: list[str] = []
    checked = 0
    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        checked += 1
        if not torch.isfinite(param).all():
            offending.append(name)
    return FiniteCheckResult(checked=checked, offending=tuple(offending))


def assert_finite_adapter(model: Any) -> None:
    """Raise `NaNWeightsError` if any trainable parameter is non-finite.

    Called from `_write_checkpoint` right after `sft.save_model()`. If
    the weights are bad, the version directory on disk still contains
    the saved (bad) adapter_model.safetensors — the trainer relies on
    `commit_version` leaving the pending dir alone when the writer
    raises, and then renames it to `{pending}-rejected` for postmortem.
    """
    result = audit_trainable_finite(model)
    if not result.ok:
        raise NaNWeightsError(list(result.offending))
