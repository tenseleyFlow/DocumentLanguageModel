"""Typed errors for the training engine.

The trainer touches a lot of heavy machinery (HF transformers, PEFT,
TRL, torch). Wrap the failure modes that have actionable user-facing
messages in typed errors so the CLI layer can map them to specific
exit codes + hints without stringly-matching arbitrary exceptions from
upstream.

Error hierarchy
---------------

- `TrainingError` — base; anything the trainer raises that the CLI
  should reword for the user.
  - `DiskSpaceError` — preflight refused to start (audit F12).
  - `OOMError` — CUDA OOM during training; carries peak/free bytes +
    the recommended `grad_accum` bump.
  - `ResumeIntegrityError` — `training_state.pt` sha256 mismatch
    (audit F12), or pinned-versions drift that we treat as fatal.
  - `VersionDriftWarning` — informational; raised (but caught at the
    trainer boundary, not the CLI) when pinned versions differ but
    the training-state is otherwise loadable.
"""

from __future__ import annotations


class TrainingError(Exception):
    """Base for `dlm.train` errors the CLI rewords for users."""


class DiskSpaceError(TrainingError):
    """Pre-flight check refused to start: free disk < estimate × margin."""

    def __init__(self, *, required_bytes: int, free_bytes: int) -> None:
        super().__init__(
            f"insufficient disk for training checkpoint: "
            f"need ~{required_bytes / 1e9:.1f} GB, have {free_bytes / 1e9:.1f} GB free"
        )
        self.required_bytes = required_bytes
        self.free_bytes = free_bytes


class OOMError(TrainingError):
    """CUDA OOM caught during training.

    `recommended_grad_accum` is a concrete next value to try; the CLI
    uses it to suggest `--grad-accum N --resume` verbatim.
    """

    def __init__(
        self,
        *,
        step: int,
        peak_bytes: int,
        free_at_start_bytes: int,
        current_grad_accum: int,
        recommended_grad_accum: int,
    ) -> None:
        super().__init__(f"CUDA OOM at step {step}")
        self.step = step
        self.peak_bytes = peak_bytes
        self.free_at_start_bytes = free_at_start_bytes
        self.current_grad_accum = current_grad_accum
        self.recommended_grad_accum = recommended_grad_accum


class ResumeIntegrityError(TrainingError):
    """Corrupted or mismatched `training_state.pt` refused on resume."""


class VersionDriftWarning(Warning):
    """Pinned versions in training_state differ from current runtime.

    Not fatal — the trainer logs an explicit diff and continues — but
    downstream consumers (the CLI, the logger) use the type to decide
    how prominently to surface the warning.
    """
