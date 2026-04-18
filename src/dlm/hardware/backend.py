"""Backend enum + lightweight detection.

`detect()` returns a `Backend` value based on the current torch build.
Kept free of heavier imports so `dlm doctor --json` on CI can tell us
"we're on CPU" without pulling in flash_attn or bitsandbytes.

The distinguishing rule for ROCm is `torch.version.hip` being truthy
(PyTorch's ROCm build sets both `torch.cuda.is_available()` True *and*
`torch.version.hip` to the HIP runtime version).
"""

from __future__ import annotations

from enum import StrEnum


class Backend(StrEnum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"
    ROCM = "rocm"


def detect() -> Backend:
    """Inspect `torch` and return the best-matching `Backend`.

    Preference order: CUDA / ROCm > MPS > CPU. This matches the order
    of least-to-most-restrictive training support.
    """
    # Lazy import: a CPU-only test run should still work if `torch` is
    # heavy to load, and we want tests to mock `torch.*` attributes via
    # the Sprint-02 hardware_mocks.
    import torch

    if torch.cuda.is_available():
        # A PyTorch ROCm build also returns True for `cuda.is_available()`
        # — the HIP runtime impersonates the CUDA surface. `torch.version.hip`
        # is the only reliable discriminator.
        if getattr(torch.version, "hip", None):
            return Backend.ROCM
        return Backend.CUDA

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return Backend.MPS

    return Backend.CPU
