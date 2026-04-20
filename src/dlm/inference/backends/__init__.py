"""Inference backends — PyTorch (default) + MLX (Apple Silicon).

Heavy backend imports are deferred. Selection logic in `select.py`
checks platform + extras without pulling in any ML framework.
"""

from __future__ import annotations

from dlm.inference.backends.base import InferenceBackend
from dlm.inference.backends.pytorch_backend import PyTorchBackend
from dlm.inference.backends.select import (
    BackendName,
    RequestedBackend,
    UnsupportedBackendError,
    build_backend,
    is_apple_silicon,
    mlx_available,
    select_backend,
)

__all__ = [
    "BackendName",
    "InferenceBackend",
    "PyTorchBackend",
    "RequestedBackend",
    "UnsupportedBackendError",
    "build_backend",
    "is_apple_silicon",
    "mlx_available",
    "select_backend",
]
