"""Capabilities probe — everything the planner needs to know about the host.

`probe()` is the single entry point that drives `dlm doctor` and
Sprint 09's training-plan resolution. Every field is a simple scalar or
a small mapping so it can be serialized to JSON without custom encoders.

Determinism class (audit F20):

- `strong`: CUDA on NVIDIA. Deterministic algorithms + cuBLAS workspace
  pinning produces bit-identical adapters across runs.
- `best-effort`: MPS. Apple's Metal kernels don't expose the same
  determinism knobs; bit-identity is not promised.
- `advisory`: CPU. Training usually deterministic, but some kernels
  (e.g., dense matmul via oneDNN) vary by CPU generation.

Telemetry posture (audit F13) reports the env-var state and whether
heavy telemetry packages (`wandb`) are importable; the `dlm` CLI sets
`HF_HUB_DISABLE_TELEMETRY=1` and `DO_NOT_TRACK=1` at entry.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Literal

import psutil

from dlm.hardware.backend import Backend, detect

DeterminismClass = Literal["strong", "best-effort", "advisory"]


@dataclass(frozen=True)
class Capabilities:
    """Snapshot of the host's relevant training-time properties."""

    backend: Backend
    device_name: str
    sm: tuple[int, int] | None
    vram_gb: float | None
    unified_memory_gb: float | None
    cpu_cores: int
    ram_gb: float
    supports_bf16: bool
    supports_fp16: bool
    has_flash_attention: bool
    has_xformers: bool
    has_bitsandbytes: bool
    has_triton: bool
    torch_version: str
    cuda_version: str | None
    rocm_version: str | None
    platform: str
    determinism_class: DeterminismClass
    telemetry_posture: dict[str, str] = field(default_factory=dict)


def probe() -> Capabilities:
    """Run every probe and return the resulting `Capabilities`.

    Pure function of current host state + installed packages. No
    side effects; safe to call from `dlm doctor --json`.
    """
    import torch

    backend = detect()
    device_name = _device_name(backend, torch)
    sm = _get_sm(backend, torch)
    vram_gb = _get_vram_gb(backend, torch)
    unified_mem = _get_unified_memory_gb(backend)
    cpu_cores = psutil.cpu_count(logical=True) or 1
    ram_gb = psutil.virtual_memory().total / (1024**3)

    return Capabilities(
        backend=backend,
        device_name=device_name,
        sm=sm,
        vram_gb=vram_gb,
        unified_memory_gb=unified_mem,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        supports_bf16=_supports_bf16(backend, sm),
        supports_fp16=_supports_fp16(backend),
        has_flash_attention=_has_flash_attention(backend, sm),
        has_xformers=_module_available("xformers"),
        has_bitsandbytes=_module_available("bitsandbytes") and backend == Backend.CUDA,
        has_triton=_module_available("triton"),
        torch_version=str(torch.__version__),
        cuda_version=_cuda_version(backend, torch),
        rocm_version=_rocm_version(torch),
        platform=_platform_string(),
        determinism_class=_determinism_class(backend),
        telemetry_posture=_telemetry_posture(),
    )


# --- probe helpers ------------------------------------------------------------


def _device_name(backend: Backend, torch: object) -> str:
    if backend in (Backend.CUDA, Backend.ROCM):
        return str(torch.cuda.get_device_name())  # type: ignore[attr-defined]
    if backend == Backend.MPS:
        return "Apple Silicon (MPS)"
    return f"CPU ({platform.processor() or platform.machine()})"


def _get_sm(backend: Backend, torch: object) -> tuple[int, int] | None:
    if backend not in (Backend.CUDA, Backend.ROCM):
        return None
    try:
        major, minor = torch.cuda.get_device_capability()  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        return None
    return (int(major), int(minor))


def _get_vram_gb(backend: Backend, torch: object) -> float | None:
    if backend not in (Backend.CUDA, Backend.ROCM):
        return None
    try:
        free, _total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        return None
    return float(free) / (1024**3)


def _get_unified_memory_gb(backend: Backend) -> float | None:
    if backend != Backend.MPS:
        return None
    # macOS aggressively pages; "free" is misleading. Report total and
    # the planner applies a 50% budget cap (see plan.py).
    return psutil.virtual_memory().total / (1024**3)


def _supports_bf16(backend: Backend, sm: tuple[int, int] | None) -> bool:
    if backend in (Backend.CUDA, Backend.ROCM):
        return sm is not None and sm >= (8, 0)
    if backend == Backend.MPS:
        # PyTorch MPS gained bf16 in 2.1; most installs support it, but
        # not every Apple SoC does. Conservative: report False and let
        # the planner pick fp16.
        return False
    return False


def _supports_fp16(backend: Backend) -> bool:
    return backend in (Backend.CUDA, Backend.ROCM, Backend.MPS)


def _has_flash_attention(backend: Backend, sm: tuple[int, int] | None) -> bool:
    """FlashAttention 2 requires NVIDIA Ampere+ and the `flash_attn` package.

    We do NOT import flash_attn here because a bad build can segfault;
    we only check that the package is present. Runtime code paths that
    use it must still wrap imports defensively.
    """
    if backend != Backend.CUDA:
        return False
    if sm is None or sm < (8, 0):
        return False
    return _module_available("flash_attn")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cuda_version(backend: Backend, torch: object) -> str | None:
    if backend != Backend.CUDA:
        return None
    return getattr(torch.version, "cuda", None)  # type: ignore[attr-defined]


def _rocm_version(torch: object) -> str | None:
    return getattr(torch.version, "hip", None)  # type: ignore[attr-defined]


def _platform_string() -> str:
    return f"{platform.system()} {platform.release()}"


def _determinism_class(backend: Backend) -> DeterminismClass:
    if backend == Backend.CUDA:
        return "strong"
    if backend == Backend.MPS:
        return "best-effort"
    if backend == Backend.ROCM:
        # ROCm deterministic kernels are a moving target; treat as
        # best-effort until Sprint 22's ROCm work lands with actual
        # measurements.
        return "best-effort"
    return "advisory"


def _telemetry_posture() -> dict[str, str]:
    """Report on env vars that gate third-party telemetry + wandb status."""
    env_vars = (
        "HF_HUB_DISABLE_TELEMETRY",
        "DO_NOT_TRACK",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS",
    )
    posture = {name: os.environ.get(name, "<unset>") for name in env_vars}
    posture["wandb_installed"] = str(_module_available("wandb"))
    posture["python"] = sys.version.split()[0]
    return posture
