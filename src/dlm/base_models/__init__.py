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
from dlm.base_models.license import (
    AcceptanceVia,
    LicenseAcceptance,
    is_gated,
    require_acceptance,
)
from dlm.base_models.registry import BASE_MODELS, known_keys
from dlm.base_models.resolver import resolve, resolve_hf
from dlm.base_models.schema import BaseModelSpec

__all__ = [
    "BASE_MODELS",
    "AcceptanceVia",
    "BaseModelError",
    "BaseModelSpec",
    "DownloadResult",
    "GatedModelError",
    "LicenseAcceptance",
    "ProbeFailedError",
    "ProbeReport",
    "ProbeResult",
    "UnknownBaseModelError",
    "download_spec",
    "is_gated",
    "known_keys",
    "require_acceptance",
    "resolve",
    "resolve_hf",
    "sha256_of_directory",
]
