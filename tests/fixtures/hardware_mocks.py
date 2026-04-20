"""Hardware-capability mocks for doctor and planner tests.

Each context manager flips a consistent set of `torch.*` attributes so
code paths gated on `torch.cuda.is_available()`, `get_device_capability()`,
`mem_get_info()`, MPS availability, or `torch.version.hip` behave as if
the requested backend is present — without real hardware.

`torch` is imported inside each function so merely collecting the module
never touches torch state.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from unittest.mock import patch


@contextlib.contextmanager
def force_cuda(
    sm: tuple[int, int] = (8, 0),
    vram_gb: float = 24.0,
    device_name: str = "NVIDIA GeForce RTX 4090",
) -> Iterator[None]:
    """Pretend a CUDA GPU with `sm` compute capability and `vram_gb` free.

    `torch.cuda.mem_get_info()` returns (free, total) in bytes; we report
    the same value for both to make arithmetic simple.
    """
    import torch

    free_bytes = int(vram_gb * (1024**3))
    total_bytes = free_bytes
    patches = [
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "device_count", return_value=1),
        patch.object(torch.cuda, "get_device_name", return_value=device_name),
        patch.object(torch.cuda, "get_device_capability", return_value=sm),
        patch.object(torch.cuda, "mem_get_info", return_value=(free_bytes, total_bytes)),
        # ROCm build attribute must be absent on a "real" CUDA box.
        patch.object(torch.version, "hip", None),
        patch.object(torch.backends.mps, "is_available", return_value=False),
        patch.object(torch.backends.mps, "is_built", return_value=False),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


@contextlib.contextmanager
def force_rocm(
    vram_gb: float = 16.0,
    device_name: str = "AMD Radeon RX 7900 XTX",
    hip_version: str = "6.0",
    sm: tuple[int, int] = (11, 0),  # HIP compute capability (RDNA3 ≈ 11.0.3)
    gcn_arch_name: str = "gfx1100",
) -> Iterator[None]:
    """Pretend a ROCm GPU. `torch.version.hip` is the distinguishing mark.

    `gcn_arch_name` (Sprint 22) is the AMD arch string — `gfx90a`
    (MI200), `gfx942` (MI300), `gfx1100` (RDNA3), `gfx1030` (RDNA2),
    etc. The bf16 + FlashAttention probes allowlist against this
    string rather than the CUDA-style SM tuple.
    """
    import torch
    from types import SimpleNamespace

    free_bytes = int(vram_gb * (1024**3))
    total_bytes = free_bytes
    device_props = SimpleNamespace(
        name=device_name,
        gcnArchName=gcn_arch_name,
        total_memory=total_bytes,
    )
    patches = [
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "device_count", return_value=1),
        patch.object(torch.cuda, "get_device_name", return_value=device_name),
        patch.object(torch.cuda, "get_device_capability", return_value=sm),
        patch.object(torch.cuda, "get_device_properties", return_value=device_props),
        patch.object(torch.cuda, "mem_get_info", return_value=(free_bytes, total_bytes)),
        patch.object(torch.version, "hip", hip_version),
        patch.object(torch.backends.mps, "is_available", return_value=False),
        patch.object(torch.backends.mps, "is_built", return_value=False),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


@contextlib.contextmanager
def force_mps() -> Iterator[None]:
    """Pretend Apple Silicon (MPS backend available, no CUDA)."""
    import torch

    patches = [
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.version, "hip", None),
        patch.object(torch.backends.mps, "is_available", return_value=True),
        patch.object(torch.backends.mps, "is_built", return_value=True),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


@contextlib.contextmanager
def force_cpu() -> Iterator[None]:
    """Pretend CPU-only (no CUDA, no MPS)."""
    import torch

    patches = [
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.version, "hip", None),
        patch.object(torch.backends.mps, "is_available", return_value=False),
        patch.object(torch.backends.mps, "is_built", return_value=False),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield
