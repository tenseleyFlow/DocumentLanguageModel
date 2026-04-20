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
    num_adapters: int = 1,
) -> None:
    """Raise `ResolutionError` on any hard-refusal combination.

    `force=True` only bypasses the CPU parameter-budget refusal, not
    QLoRA-on-MPS or similar which would produce outright failures.

    `num_adapters` is the count of declared adapters (``len(training.adapters)``
    or 1 for single-adapter docs). When >1 and QLoRA is requested, F28
    refuses because the combined adapter states + base would typically
    blow the VRAM budget.
    """
    adapter_choice = _effective_adapter(training)

    if adapter_choice == "qlora":
        _refuse_qlora(caps)

    if (
        adapter_choice == "qlora"
        and num_adapters > 1
        and caps.backend == Backend.CUDA
        and caps.vram_gb is not None
    ):
        # F28: coarse VRAM estimate. QLoRA base at 4-bit ≈ params * 0.5 bytes;
        # each named adapter carries its own LoRA + optimizer state (~1 GB
        # worst-case on the adapter sizes we ship). Plus a 25% activation
        # overhead. When the sum exceeds the 85%-of-VRAM headroom, refuse.
        base_gb = base_params * 0.5 / 1e9  # 4-bit base
        per_adapter_gb = 1.0
        activations_gb = base_params * 2.0 / 1e9 * 0.25
        est_peak = base_gb + per_adapter_gb * num_adapters + activations_gb
        budget = caps.vram_gb * 0.85
        if est_peak > budget:
            raise ResolutionError(
                "Multi-adapter QLoRA would exceed VRAM "
                f"(~{est_peak:.1f} GB estimated vs {budget:.1f} GB budget "
                f"for {caps.vram_gb:.0f} GB device); "
                "try `adapter: lora` instead of `qlora`, or reduce the "
                "number of adapters.",
            )

    if caps.backend == Backend.CPU and base_params > CPU_PARAM_BUDGET and not force:
        raise ResolutionError(
            f"Refusing to train a {base_params / 1e6:.0f}M-parameter model on CPU. "
            "CPU training is capped at 200M parameters by default; pass "
            "`--force` if you accept very slow steps.",
        )


def _effective_adapter(training: TrainingConfig) -> str:
    """Return the adapter type effectively in force.

    Single-adapter docs carry the choice at the top level. Multi-adapter
    docs declare it per-adapter; for refusal purposes we take "qlora if
    any adapter is qlora" — a mixed multi-adapter doc still needs to
    satisfy the stricter constraint.
    """
    if training.adapters is None:
        return training.adapter
    if any(a.adapter == "qlora" for a in training.adapters.values()):
        return "qlora"
    return "lora"


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
