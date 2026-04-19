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

from dlm.base_models.downloader import DownloadResult, download_spec, sha256_of_directory
from dlm.base_models.errors import (
    BaseModelError,
    GatedModelError,
    ProbeFailedError,
    ProbeReport,
    ProbeResult,
    UnknownBaseModelError,
)
from dlm.base_models.registry import BASE_MODELS, known_keys
from dlm.base_models.resolver import resolve, resolve_hf
from dlm.base_models.schema import BaseModelSpec

__all__ = [
    "BASE_MODELS",
    "BaseModelError",
    "BaseModelSpec",
    "DownloadResult",
    "GatedModelError",
    "ProbeFailedError",
    "ProbeReport",
    "ProbeResult",
    "UnknownBaseModelError",
    "download_spec",
    "known_keys",
    "resolve",
    "resolve_hf",
    "sha256_of_directory",
]
