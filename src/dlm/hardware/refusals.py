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
        # F28: estimate peak VRAM for the multi-adapter + QLoRA case.
        # Base lives once in VRAM at 4-bit. Each adapter carries its own
        # LoRA params + AdamW state + gradients, scaling with `base_params`
        # and the adapter's `lora_r` (our LoRA params ≈ 2 * base * r / hidden;
        # AdamW state ≈ 2×LoRA params in fp32). A 7B QLoRA r=16 adapter
        # lands around 300-500 MB; a 135M r=8 adapter is ~10 MB. A flat
        # 1 GB/adapter (pre-audit) was 30× too high for small bases and
        # 2× too low for large ones. The formula below scales linearly
        # with `avg_lora_r × base_params`; 0.1 GB floor keeps tiny
        # multi-adapter setups from false-greenlighting.
        #
        # Multi-GPU note: DDP replicates the model across ranks (each
        # GPU holds the full base + adapter state), so
        # `world_size` does NOT divide the per-GPU VRAM math. The
        # formula stays conservative when scaled by rank count — we
        # never want to greenlight a config that fits on N GPUs
        # individually but not on any single one. Sharded /
        # FSDP / ZeRO-3 paths would need a different calculation
        # (they're out of scope here).
        avg_lora_r = _avg_lora_r(training)
        base_gb = base_params * 0.5 / 1e9  # 4-bit base
        per_adapter_gb = max(0.1, base_params * avg_lora_r / (1e9 * 64))
        activations_gb = base_params * 2.0 / 1e9 * 0.25
        qlora_adapter_count = _qlora_adapter_count(training, num_adapters)
        est_peak = base_gb + per_adapter_gb * qlora_adapter_count + activations_gb
        budget = caps.vram_gb * 0.85
        if est_peak > budget:
            offenders = _qlora_adapter_names(training)
            offender_note = f" (offending adapters: {sorted(offenders)})" if offenders else ""
            raise ResolutionError(
                "Multi-adapter QLoRA would exceed VRAM "
                f"(~{est_peak:.1f} GB estimated vs {budget:.1f} GB budget "
                f"for {caps.vram_gb:.0f} GB device){offender_note}; "
                "try `adapter: lora` instead of `qlora`, or reduce the "
                "number of adapters.",
            )

    if caps.backend == Backend.CPU and base_params > CPU_PARAM_BUDGET and not force:
        raise ResolutionError(
            f"Refusing to train a {base_params / 1e6:.0f}M-parameter model on CPU. "
            "CPU training is capped at 200M parameters by default; pass "
            "`--force` if you accept very slow steps.",
        )


def check_multi_gpu_refusals(caps: Capabilities, world_size: int) -> None:
    """Refuse multi-GPU configurations that can't reasonably work.

    Current multi-GPU support is CUDA only. MPS doesn't do DDP; CPU
    multi-process training is technically possible but a terrible user
    experience. Heterogeneous CUDA GPUs (different SM families)
    produce inconsistent mixed-precision results — refuse rather than
    let the slower arch silently dictate the precision.

    ROCm multi-GPU is explicitly out of scope for now — refuse with a
    pointer so users don't chase phantom bugs.
    """
    if world_size < 2:
        return
    if caps.backend == Backend.MPS:
        raise ResolutionError(
            "Multi-GPU training on Apple Silicon (MPS) is not supported; "
            "MPS has no DDP path. Train single-GPU or on a CUDA host.",
        )
    if caps.backend == Backend.CPU:
        raise ResolutionError(
            "Multi-GPU training on CPU is not supported. Drop `--gpus` or run single-process.",
        )
    if caps.backend == Backend.ROCM:
        raise ResolutionError(
            "Multi-GPU training on ROCm is not supported yet; "
            "train single-GPU on ROCm or use a CUDA host for multi-GPU runs.",
        )
    # CUDA path — heterogeneous detection is the caller's responsibility
    # since `Capabilities` only reports a single device. Callers that
    # assemble multi-device state (the launcher) should call
    # `assert_homogeneous_cuda` directly before spawning ranks.


def assert_homogeneous_cuda(sm_per_device: list[tuple[int, int] | None]) -> None:
    """Refuse if the configured CUDA devices span different SM families.

    Accepts the list of SM tuples the launcher collected from
    `torch.cuda.get_device_capability(i)` for each selected device.
    Mixed precision behavior on heterogeneous GPUs (e.g. Ampere +
    Turing) is unreliable — bf16 paths silently fall back to fp16 on
    the Turing card and the two ranks drift.
    """
    unique = {sm for sm in sm_per_device if sm is not None}
    if len(unique) > 1:
        raise ResolutionError(
            f"Heterogeneous CUDA GPUs detected (SM families: {sorted(unique)}); "
            "multi-GPU training requires matching compute capability. "
            "Select GPUs of the same generation via `--gpus 0,1` etc.",
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


def _avg_lora_r(training: TrainingConfig) -> float:
    """Average LoRA rank across declared adapters (fallback: flat lora_r)."""
    if training.adapters is None or not training.adapters:
        return float(training.lora_r)
    return sum(a.lora_r for a in training.adapters.values()) / len(training.adapters)


def _qlora_adapter_count(training: TrainingConfig, fallback: int) -> int:
    """Return the count of QLoRA-typed adapters; `fallback` for flat docs."""
    if training.adapters is None:
        return fallback
    return sum(1 for a in training.adapters.values() if a.adapter == "qlora")


def _qlora_adapter_names(training: TrainingConfig) -> list[str]:
    """Return the declared adapter names using QLoRA (empty on flat docs)."""
    if training.adapters is None:
        return []
    return [n for n, a in training.adapters.items() if a.adapter == "qlora"]


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
            "QLoRA on ROCm is not supported: `bitsandbytes` ROCm builds are "
            "upstream-unstable and we refuse rather than silently corrupt "
            "weights. LoRA on ROCm works — set `adapter: lora`.",
        )
    if not caps.has_bitsandbytes:
        raise ResolutionError(
            "QLoRA requires `bitsandbytes`; not found in the current environment. "
            "Install with `uv sync --extra cuda` (once the extra lands) or set "
            "`adapter: lora`.",
        )
