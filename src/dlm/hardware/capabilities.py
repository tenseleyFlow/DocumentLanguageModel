"""Capabilities probe — everything the planner needs to know about the host.

`probe()` is the single entry point that drives `dlm doctor` and
training-plan resolution. Every field is a simple scalar or a small
mapping so it can be serialized to JSON without custom encoders.

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
import logging
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Literal

import psutil

from dlm.hardware.backend import Backend, detect

_LOG = logging.getLogger(__name__)

DeterminismClass = Literal["strong", "best-effort", "advisory"]


@dataclass(frozen=True)
class Capabilities:
    """Snapshot of the host's relevant training-time properties."""

    backend: Backend
    device_name: str
    sm: tuple[int, int] | None
    rocm_arch: str | None
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
    has_mlx: bool
    torch_version: str
    accelerate_version: str | None
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
    rocm_arch = _get_rocm_arch(backend, torch)
    vram_gb = _get_vram_gb(backend, torch)
    unified_mem = _get_unified_memory_gb(backend)
    cpu_cores = psutil.cpu_count(logical=True) or 1
    ram_gb = psutil.virtual_memory().total / (1024**3)

    return Capabilities(
        backend=backend,
        device_name=device_name,
        sm=sm,
        rocm_arch=rocm_arch,
        vram_gb=vram_gb,
        unified_memory_gb=unified_mem,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        supports_bf16=_supports_bf16(backend, sm, rocm_arch),
        supports_fp16=_supports_fp16(backend),
        has_flash_attention=_has_flash_attention(backend, sm, rocm_arch),
        has_xformers=_module_available("xformers"),
        has_bitsandbytes=_module_available("bitsandbytes") and backend == Backend.CUDA,
        has_triton=_module_available("triton"),
        has_mlx=_has_mlx_inference(backend),
        torch_version=str(torch.__version__),
        accelerate_version=_accelerate_version(),
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
    except (AttributeError, RuntimeError) as exc:
        _LOG.warning(
            "capabilities probe: torch.cuda.get_device_capability failed (%s); "
            "treating SM as unknown",
            exc,
        )
        return None
    return (int(major), int(minor))


def _get_vram_gb(backend: Backend, torch: object) -> float | None:
    if backend not in (Backend.CUDA, Backend.ROCM):
        return None
    try:
        free, _total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError) as exc:
        _LOG.warning(
            "capabilities probe: torch.cuda.mem_get_info failed (%s); treating VRAM as unknown",
            exc,
        )
        return None
    return float(free) / (1024**3)


def _get_unified_memory_gb(backend: Backend) -> float | None:
    if backend != Backend.MPS:
        return None
    # macOS aggressively pages; "free" is misleading. Report total and
    # the planner applies a 50% budget cap (see plan.py).
    return psutil.virtual_memory().total / (1024**3)


def _supports_bf16(backend: Backend, sm: tuple[int, int] | None, rocm_arch: str | None) -> bool:
    if backend == Backend.CUDA:
        return sm is not None and sm >= (8, 0)
    if backend == Backend.ROCM:
        return _rocm_arch_supports_bf16(rocm_arch)
    if backend == Backend.MPS:
        # PyTorch MPS gained bf16 in 2.1; most installs support it, but
        # not every Apple SoC does. Conservative: report False and let
        # the planner pick fp16.
        return False
    return False


def _supports_fp16(backend: Backend) -> bool:
    return backend in (Backend.CUDA, Backend.ROCM, Backend.MPS)


def _has_flash_attention(
    backend: Backend, sm: tuple[int, int] | None, rocm_arch: str | None
) -> bool:
    """FlashAttention 2 availability by backend + arch.

    - NVIDIA: `flash_attn` + SM ≥ 8.0 (Ampere+).
    - ROCm: AMD's `flash_attn` fork exists for CDNA
      (gfx90a MI200, gfx942 MI300) and is landing for RDNA3. We gate
      on the same bf16-capable arch allowlist — if an arch is too old
      for bf16 it can't run FA2 either — and on the `flash_attn`
      package being importable. We do NOT import it here because a
      bad build can segfault; runtime paths must wrap imports
      defensively.
    """
    if not _module_available("flash_attn"):
        return False
    if backend == Backend.CUDA:
        return sm is not None and sm >= (8, 0)
    if backend == Backend.ROCM:
        return _rocm_arch_supports_bf16(rocm_arch)
    return False


# --- ROCm arch helpers -------------------------------------------------------


def _get_rocm_arch(backend: Backend, torch: object) -> str | None:
    """Read `gcnArchName` off the first ROCm device; None off-ROCm.

    `gcnArchName` is the canonical AMD arch string (`gfx90a`,
    `gfx942`, `gfx1100`, ...). Distinct from the CUDA-style SM tuple
    on ROCm because AMD arches don't compose as (major, minor) — they
    map to discrete micro-architectures with different ISA surfaces.
    """
    if backend != Backend.ROCM:
        return None
    try:
        props = torch.cuda.get_device_properties(0)  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError, AssertionError) as exc:
        _LOG.warning(
            "capabilities probe: torch.cuda.get_device_properties failed on ROCm (%s); "
            "treating arch as unknown",
            exc,
        )
        return None
    name = getattr(props, "gcnArchName", None)
    if not isinstance(name, str) or not name:
        return None
    # Some ROCm builds append a colon-separated xnack/sramecc suffix
    # (e.g. `gfx90a:sramecc+:xnack-`). Strip it so the allowlist can
    # match exact arches.
    return name.split(":", 1)[0]


def _rocm_arch_supports_bf16(arch: str | None) -> bool:
    """ROCm arches with real bf16 (not emulated) — CDNA2+ and RDNA3+.

    - `gfx90a` — MI200 (CDNA2)
    - `gfx942` — MI300 (CDNA3)
    - `gfx1100`/`gfx1101`/`gfx1102` — RDNA3 (7900 XTX, 7800 XT, 7700 XT)
    - `gfx1200`/`gfx1201` — RDNA4
    Older arches (gfx906 Vega20, gfx908 MI100 CDNA1, gfx1010-gfx1036
    RDNA1/RDNA2) either lack bf16 or emulate it at a cost that defeats
    the purpose.
    """
    if arch is None:
        return False
    allowlist = {"gfx90a", "gfx942", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"}
    return arch in allowlist


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_mlx_inference(backend: Backend) -> bool:
    """True iff MLX inference is runnable on this host.

    MLX is darwin-arm64 only. Off-platform installs of `mlx` via pip
    would be a packaging mistake, but we still gate on backend to
    avoid reporting True for a misconfigured CUDA box that happens to
    have an mlx dist lying around.
    """
    if backend != Backend.MPS:
        return False
    return _module_available("mlx") and _module_available("mlx_lm")


def _cuda_version(backend: Backend, torch: object) -> str | None:
    if backend != Backend.CUDA:
        return None
    return getattr(torch.version, "cuda", None)  # type: ignore[attr-defined]


def _rocm_version(torch: object) -> str | None:
    return getattr(torch.version, "hip", None)  # type: ignore[attr-defined]


def _accelerate_version() -> str | None:
    """Read `accelerate.__version__` without importing the heavy side.

    The doctor reports accelerate so users can tell whether multi-GPU
    (`dlm train --gpus`) is available. Absent install → None; doctor
    renders "not installed".
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version("accelerate")
    except PackageNotFoundError:
        return None


def _platform_string() -> str:
    return f"{platform.system()} {platform.release()}"


def _determinism_class(backend: Backend) -> DeterminismClass:
    if backend == Backend.CUDA:
        return "strong"
    if backend == Backend.MPS:
        return "best-effort"
    if backend == Backend.ROCM:
        # ROCm deterministic kernels are a moving target across HIP
        # versions; MIOpen's deterministic conv path exists but
        # determinism across full training is not guaranteed the way
        # it is on CUDA with `use_deterministic_algorithms`. Classified
        # "best-effort" — fp match may drift across PyTorch/ROCm
        # upgrades even with a pinned seed.
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
