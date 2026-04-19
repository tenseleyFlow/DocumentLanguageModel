"""Curated base-model registry + `hf:org/name` escape hatch.

Public surface:

- `BASE_MODELS`, `known_keys()` — registry dict + stable ordering
- `BaseModelSpec` — per-entry metadata (frozen, strict)
- `resolve(spec)` — `key` or `hf:org/name` → `BaseModelSpec`
- Probes live in `dlm.base_models.probes`; downloader in `downloader.py`
- Errors: `BaseModelError`, `UnknownBaseModelError`, `ProbeFailedError`,
  `GatedModelError`
"""

from __future__ import annotations

from dlm.base_models.errors import (
    BaseModelError,
    GatedModelError,
    ProbeFailedError,
    ProbeReport,
    ProbeResult,
    UnknownBaseModelError,
)
from dlm.base_models.registry import BASE_MODELS, known_keys
from dlm.base_models.schema import BaseModelSpec

__all__ = [
    "BASE_MODELS",
    "BaseModelError",
    "BaseModelSpec",
    "GatedModelError",
    "ProbeFailedError",
    "ProbeReport",
    "ProbeResult",
    "UnknownBaseModelError",
    "known_keys",
]
