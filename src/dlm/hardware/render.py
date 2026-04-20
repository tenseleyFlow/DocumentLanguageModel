"""Human-readable renderer for `dlm doctor` output.

Kept separate from `doctor.py` so Sprint 13's CLI can theme the output
with `rich` if desired without the core module pulling rich in.
"""

from __future__ import annotations

from dlm.hardware.capabilities import Capabilities
from dlm.hardware.doctor import DoctorResult
from dlm.hardware.plan import TrainingPlan

_PLAN_REFERENCE_NOTE = "1.5B params, seq_len 2048"


def render_text(result: DoctorResult) -> str:
    """Produce the plain-text block shown by `dlm doctor`."""
    caps = result.capabilities
    lines: list[str] = []

    # Capability block
    lines.append(f"Backend:        {_backend_line(caps)}")
    lines.append(f"Torch:          {caps.torch_version}{_torch_suffix(caps)}")
    lines.append(f"bitsandbytes:   {_bool(caps.has_bitsandbytes)}")
    lines.append(f"FlashAttention: {_bool(caps.has_flash_attention)}")
    lines.append(f"xFormers:       {_bool(caps.has_xformers)}")
    lines.append(f"Triton:         {_bool(caps.has_triton)}")
    lines.append(f"MLX inference:  {_bool(caps.has_mlx)}")
    lines.append(f"CPU cores:      {caps.cpu_cores}")
    lines.append(f"RAM:            {caps.ram_gb:.1f} GB")
    lines.append(f"Determinism:    {caps.determinism_class}")
    lines.append(f"Telemetry:      {_telemetry_line(caps.telemetry_posture)}")
    lines.append(f"Platform:       {caps.platform}")

    if result.plan is not None:
        lines.append("")
        lines.append(f"Suggested plan (reference base, {_PLAN_REFERENCE_NOTE}):")
        plan = result.plan
        lines.append(f"  adapter:         {_adapter_summary(plan)}")
        lines.append(f"  precision:       {plan.precision}")
        lines.append(f"  attention:       {plan.attn_implementation}")
        lines.append(
            f"  micro_batch:     {plan.micro_batch_size}  (grad_accum {plan.grad_accum}, "
            f"effective {plan.effective_batch_size})"
        )
        lines.append(f"  grad_ckpt:       {_bool(plan.gradient_checkpointing)}")
        lines.append(f"  est. peak VRAM:  {plan.est_peak_vram_gb:.2f} GB")
        lines.append(f"  est. step time:  ~{plan.est_step_seconds:.2f} s")
        lines.append(f"  reason:          {plan.reason}")
    elif result.plan_error:
        lines.append("")
        lines.append(f"Plan refused: {result.plan_error}")

    lines.append("")
    lines.append("Run `dlm doctor --json` for machine-readable output.")
    return "\n".join(lines)


def _backend_line(caps: Capabilities) -> str:
    bits: list[str] = [f"{caps.backend.value} ({caps.device_name})"]
    if caps.sm is not None:
        bits.append(f"SM {caps.sm[0]}.{caps.sm[1]}")
    if caps.vram_gb is not None:
        bits.append(f"{caps.vram_gb:.1f} GB VRAM free")
    if caps.unified_memory_gb is not None:
        bits.append(f"{caps.unified_memory_gb:.1f} GB unified mem")
    return ", ".join(bits)


def _torch_suffix(caps: Capabilities) -> str:
    if caps.cuda_version:
        return f" (CUDA {caps.cuda_version})"
    if caps.rocm_version:
        return f" (ROCm {caps.rocm_version})"
    return ""


def _bool(v: bool) -> str:
    return "yes" if v else "no"


def _telemetry_line(posture: dict[str, str]) -> str:
    hf = posture.get("HF_HUB_DISABLE_TELEMETRY", "<unset>")
    dnt = posture.get("DO_NOT_TRACK", "<unset>")
    wandb = posture.get("wandb_installed", "False")
    disabled = hf == "1" and dnt == "1"
    status = "disabled" if disabled else "PARTIAL — check env vars"
    return f"{status} (HF={hf}, DO_NOT_TRACK={dnt}, wandb_installed={wandb})"


def _adapter_summary(plan: TrainingPlan) -> str:
    if plan.use_qlora:
        return f"qlora (4-bit NF4, compute {plan.quant_compute_dtype})"
    return "lora"
