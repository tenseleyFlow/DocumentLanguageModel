"""Training-plan resolver.

`resolve()` takes a `TrainingConfig` (frontmatter) + `Capabilities` (host
probe) + base-model size, calls the refusal matrix, then picks concrete
values for `auto` fields:

- `precision`: bf16 on Ampere+ CUDA; fp16 on MPS / older CUDA / ROCm.
- `attn_implementation`: FlashAttention 2 when available, SDPA otherwise.
- `use_qlora`: True iff user requested it AND the backend supports it.
- `micro_batch_size`: 1 by default; grown when memory budget permits.
- `grad_accum`: chosen to reach a target effective batch (default 8).
- `gradient_checkpointing`: True when estimated peak VRAM would otherwise
  exceed 70% of the memory budget.

The returned `TrainingPlan` is frozen and JSON-serializable.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

from dlm.doc.schema import TrainingConfig
from dlm.hardware.backend import Backend
from dlm.hardware.capabilities import Capabilities
from dlm.hardware.memory import estimate_peak_vram_gb, estimate_step_seconds
from dlm.hardware.refusals import check_multi_gpu_refusals, check_refusals

_LOG = logging.getLogger(__name__)

AttnImpl = Literal["flash_attention_2", "sdpa", "eager"]
Precision = Literal["bf16", "fp16", "fp32"]

DEFAULT_EFFECTIVE_BATCH: Final[int] = 8
MPS_BUDGET_FRACTION: Final[float] = 0.50  # half of unified memory
CUDA_BUDGET_FRACTION: Final[float] = 0.85  # leave ~15% headroom on CUDA


@dataclass(frozen=True)
class TrainingPlan:
    """Resolved training plan for the current host.

    Fields mirror the knobs the trainer actually consumes.
    `world_size` is the number of data-parallel ranks; 1 on
    single-GPU / single-process paths. `effective_batch_size` already
    folds `world_size` in, so users reading the plan don't have to
    multiply themselves.
    """

    precision: Precision
    attn_implementation: AttnImpl
    use_qlora: bool
    quant_compute_dtype: str | None  # bnb_4bit_compute_dtype, None iff not QLoRA
    micro_batch_size: int
    grad_accum: int
    effective_batch_size: int
    gradient_checkpointing: bool
    est_peak_vram_gb: float
    est_step_seconds: float
    reason: str
    world_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view for `dlm doctor --json`."""
        return asdict(self)


Phase = Literal["sft", "dpo"]


def resolve(
    training: TrainingConfig,
    caps: Capabilities,
    *,
    base_params: int,
    seq_len: int,
    force: bool = False,
    phase: Phase = "sft",
    num_adapters: int = 1,
    world_size: int = 1,
) -> TrainingPlan:
    """Produce a concrete plan from a frontmatter config + host caps.

    `phase="dpo"` halves the resolved micro-batch to account for the
    policy + reference forward passes DPO runs per step. Grad-accum
    stays; the effective batch halves with the micro-batch. Memory
    and step-time estimates are recomputed against the adjusted
    micro-batch.

    `num_adapters` lets multi-adapter callers surface the count so the
    multi-adapter QLoRA VRAM refusal can fire before training starts.
    Single-adapter docs keep the default.

    `world_size` is the number of data-parallel ranks. Multiplies the
    reported `effective_batch_size` (each rank processes a
    micro-batch independently) and scales the per-rank step-time
    estimate down — more GPUs, less wall-clock time per global step
    up to comm overhead. `world_size > 1` triggers the multi-GPU
    refusal matrix (MPS/CPU refusal, heterogeneous CUDA refusal).
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if world_size > 1:
        check_multi_gpu_refusals(caps, world_size)
    check_refusals(training, caps, base_params, force=force, num_adapters=num_adapters)

    use_qlora = _should_qlora(training, caps)
    precision = _pick_precision(caps, override=training.precision)
    attn = _pick_attention(caps)
    micro_batch = _resolve_micro_batch(training, caps, base_params, seq_len, precision, use_qlora)
    if phase == "dpo":
        micro_batch = max(1, micro_batch // 2)
    grad_accum = _resolve_grad_accum(training, micro_batch)
    gradient_checkpointing = _needs_gradient_checkpointing(
        caps, base_params, seq_len, micro_batch, precision, use_qlora, training.lora_r
    )
    est_peak = estimate_peak_vram_gb(
        base_params=base_params,
        seq_len=seq_len,
        micro_batch=micro_batch,
        precision=precision,
        use_qlora=use_qlora,
        lora_r=training.lora_r,
        gradient_checkpointing=gradient_checkpointing,
        optimizer=training.optimizer,
    )
    if phase == "dpo":
        # Policy + frozen reference both live in VRAM; rough ×2 on
        # model-weight share of peak usage.
        est_peak *= 1.8
    est_step = estimate_step_seconds(
        backend=caps.backend,
        base_params=base_params,
        seq_len=seq_len,
        micro_batch=micro_batch,
        has_flash_attention=attn == "flash_attention_2",
    )
    if phase == "dpo":
        # Two forward passes per step plus the pair-loss backward.
        est_step *= 2.2

    reason = _build_reason(
        precision,
        attn,
        use_qlora,
        gradient_checkpointing,
        phase=phase,
        adapter=training.adapter,
        optimizer=training.optimizer,
        base_params=base_params,
    )
    quant_dtype = precision if use_qlora else None

    return TrainingPlan(
        precision=precision,
        attn_implementation=attn,
        use_qlora=use_qlora,
        quant_compute_dtype=quant_dtype,
        micro_batch_size=micro_batch,
        grad_accum=grad_accum,
        effective_batch_size=micro_batch * grad_accum * world_size,
        gradient_checkpointing=gradient_checkpointing,
        world_size=world_size,
        est_peak_vram_gb=round(est_peak, 2),
        est_step_seconds=round(est_step, 2),
        reason=reason,
    )


# --- resolver helpers ---------------------------------------------------------


def _should_qlora(training: TrainingConfig, caps: Capabilities) -> bool:
    return training.adapter == "qlora" and caps.backend == Backend.CUDA and caps.has_bitsandbytes


def _pick_precision(caps: Capabilities, override: Precision | None = None) -> Precision:
    """Pick training precision.

    Defaults:

    - CUDA Ampere+ / ROCm with bf16 support → bf16
    - Older CUDA → fp16
    - MPS → fp32 (pinned: PyTorch's MPS fp16 attention kernels produce
      NaN LoRA weights on tiny-data SFT runs; reproduced 2026-04-20 on
      smollm2-135m + 4 samples + lr=2e-4 + no warmup, see
      `.docs/bugs/01-nan-adapter-on-mps.md`)
    - CPU → fp32 implicit (memory estimator treats it as full precision)

    An explicit `override` from `training.precision` wins. On MPS, an
    fp16 override emits a loud warning — the user is opting into the
    known NaN risk in exchange for fitting larger bases in unified
    memory. A bf16 override is silently honored where supported; on
    hosts that don't support bf16 the caller should refuse at an
    earlier boundary (this function trusts its inputs).
    """
    if override is not None:
        if caps.backend == Backend.MPS and override == "fp16":
            _LOG.warning(
                "training.precision=fp16 on MPS: accepting user override, but "
                "the MPS fp16 attention kernels can produce NaN LoRA weights "
                "on tiny-data runs. The post-train finite-weight gate will "
                "refuse to persist a corrupt adapter. If training diverges, "
                "drop back to fp32 (the default) or add warmup + more data. "
                "See .docs/bugs/01-nan-adapter-on-mps.md."
            )
        return override
    if caps.backend == Backend.MPS:
        return "fp32"
    if caps.supports_bf16:
        return "bf16"
    return "fp16"


def _pick_attention(caps: Capabilities) -> AttnImpl:
    if caps.has_flash_attention:
        return "flash_attention_2"
    if caps.backend == Backend.CPU:
        # SDPA can be slow on CPU; eager is often more predictable.
        return "eager"
    return "sdpa"


def _memory_budget_gb(caps: Capabilities) -> float:
    """How much memory the planner is willing to use."""
    if caps.backend in (Backend.CUDA, Backend.ROCM) and caps.vram_gb is not None:
        return caps.vram_gb * CUDA_BUDGET_FRACTION
    if caps.backend == Backend.MPS and caps.unified_memory_gb is not None:
        return caps.unified_memory_gb * MPS_BUDGET_FRACTION
    # CPU: advisory; we use system RAM as the ceiling.
    return caps.ram_gb * 0.5


def _resolve_micro_batch(
    training: TrainingConfig,
    caps: Capabilities,
    base_params: int,
    seq_len: int,
    precision: Precision,
    use_qlora: bool,
) -> int:
    if training.micro_batch_size != "auto":
        return int(training.micro_batch_size)

    budget_gb = _memory_budget_gb(caps)
    # Start at 1 and grow as long as the estimate fits.
    chosen = 1
    for candidate in (2, 4, 8, 16):
        est = estimate_peak_vram_gb(
            base_params=base_params,
            seq_len=seq_len,
            micro_batch=candidate,
            precision=precision,
            use_qlora=use_qlora,
            lora_r=training.lora_r,
            gradient_checkpointing=True,
            optimizer=training.optimizer,
        )
        if est > budget_gb:
            break
        chosen = candidate
    return chosen


def _resolve_grad_accum(training: TrainingConfig, micro_batch: int) -> int:
    if training.grad_accum != "auto":
        return int(training.grad_accum)
    # Choose grad_accum to hit DEFAULT_EFFECTIVE_BATCH, rounding up.
    if micro_batch >= DEFAULT_EFFECTIVE_BATCH:
        return 1
    return max(1, DEFAULT_EFFECTIVE_BATCH // micro_batch)


def _needs_gradient_checkpointing(
    caps: Capabilities,
    base_params: int,
    seq_len: int,
    micro_batch: int,
    precision: Precision,
    use_qlora: bool,
    lora_r: int,
) -> bool:
    budget = _memory_budget_gb(caps)
    est_no_ckpt = estimate_peak_vram_gb(
        base_params=base_params,
        seq_len=seq_len,
        micro_batch=micro_batch,
        precision=precision,
        use_qlora=use_qlora,
        lora_r=lora_r,
        gradient_checkpointing=False,
    )
    return est_no_ckpt > (budget * 0.70)


_GALORE_SMALL_BASE_THRESHOLD: Final = 1_000_000_000


def _build_reason(
    precision: Precision,
    attn: AttnImpl,
    use_qlora: bool,
    gradient_checkpointing: bool,
    *,
    phase: Phase = "sft",
    adapter: str = "lora",
    optimizer: str = "adamw_torch",
    base_params: int = 0,
) -> str:
    parts = [f"precision={precision}", f"attn={attn}"]
    if use_qlora:
        parts.append("qlora=4bit-nf4")
    else:
        parts.append("qlora=off")
    if adapter == "dora":
        parts.append("adapter=dora")
    if optimizer.startswith("galore_"):
        parts.append(f"optim={optimizer}")
        # Soft warning: the GaLore paper reports uplift at ≥ 7B; below
        # ~1B the optimizer memory savings are real but the quality
        # impact is frequently negative-to-neutral. Surface the caveat
        # in the plan rather than blocking — operators may still want
        # GaLore for its memory footprint on a constrained host.
        if 0 < base_params < _GALORE_SMALL_BASE_THRESHOLD:
            parts.append(f"warn=galore-small-base({base_params / 1e6:.0f}M<1B)")
    if gradient_checkpointing:
        parts.append("grad_ckpt=on")
    if phase == "dpo":
        parts.append("phase=dpo")
    return ", ".join(parts)
