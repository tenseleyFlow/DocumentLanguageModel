"""Refusal matrix — invalid training configurations rejected early.

Per findings §9 and audit F24, we refuse training configurations that
would either silently produce garbage or blow up at runtime after hours
of waiting.

Behavior:

- Hard refusals raise `ResolutionError` with an actionable message.
- Soft-refusal ("warn-only") situations — like FlashAttention requested
  on CPU — live in `plan.resolve()` which downgrades the request to a
  safe fallback (SDPA) and records the rationale in the plan.

The CPU parameter threshold is **>200M** (not >3B) per the overview
hardware matrix and audit F24's consistency fix.
"""

from __future__ import annotations

from typing import Final

from dlm.doc.schema import TrainingConfig
from dlm.hardware.backend import Backend
from dlm.hardware.capabilities import Capabilities

CPU_PARAM_BUDGET: Final[int] = 200_000_000  # audit F24


class ResolutionError(Exception):
    """A requested training configuration cannot be honored on this host."""


def check_refusals(
    training: TrainingConfig,
    caps: Capabilities,
    base_params: int,
    *,
    force: bool = False,
) -> None:
    """Raise `ResolutionError` on any hard-refusal combination.

    `force=True` only bypasses the CPU parameter-budget refusal, not
    QLoRA-on-MPS or similar which would produce outright failures.
    """
    if training.adapter == "qlora":
        _refuse_qlora(caps)

    if caps.backend == Backend.CPU and base_params > CPU_PARAM_BUDGET and not force:
        raise ResolutionError(
            f"Refusing to train a {base_params / 1e6:.0f}M-parameter model on CPU. "
            "CPU training is capped at 200M parameters by default; pass "
            "`--force` if you accept very slow steps.",
        )


def _refuse_qlora(caps: Capabilities) -> None:
    if caps.backend == Backend.MPS:
        raise ResolutionError(
            "QLoRA requires CUDA + bitsandbytes; MPS detected. "
            "Set `adapter: lora` in the frontmatter or train on a CUDA box.",
        )
    if caps.backend == Backend.CPU:
        raise ResolutionError(
            "QLoRA requires CUDA + bitsandbytes; CPU detected. "
            "Set `adapter: lora` in the frontmatter.",
        )
    if caps.backend == Backend.ROCM:
        raise ResolutionError(
            "QLoRA on ROCm is not supported (bitsandbytes ROCm is unstable). "
            "Set `adapter: lora`, or wait for Sprint 22's ROCm work.",
        )
    if not caps.has_bitsandbytes:
        raise ResolutionError(
            "QLoRA requires `bitsandbytes`; not found in the current environment. "
            "Install with `uv sync --extra cuda` (once the extra lands) or set "
            "`adapter: lora`.",
        )
