"""Inference backends — PyTorch (default) + MLX (Apple Silicon).

Heavy backend imports are deferred. Selection logic in `select.py`
checks platform + extras without pulling in any ML framework.
"""

from __future__ import annotations

from dlm.inference.backends.base import InferenceBackend
from dlm.inference.backends.pytorch_backend import PyTorchBackend

__all__ = ["InferenceBackend", "PyTorchBackend"]
