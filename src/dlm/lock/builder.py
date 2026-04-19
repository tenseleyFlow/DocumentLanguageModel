"""Construct a `DlmLock` from run context (Sprint 15).

Factors the lock-assembly away from the trainer / export runner so
both call-sites share one implementation. Keeps the trainer focused
on training orchestration.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from dlm.lock.schema import DeterminismClass, DlmLock, HardwareTier

if TYPE_CHECKING:
    from dlm.base_models.license import LicenseAcceptance


def hash_dlm_file(path: Path) -> str:
    """SHA-256 hex of the `.dlm` file bytes (raw, pre-parse)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


_BACKEND_TO_TIER: Final[dict[str, HardwareTier]] = {
    "cpu": "cpu",
    "mps": "mps",
    "rocm": "rocm",
}


def hardware_tier_from_backend(
    backend: str | None,
    *,
    sm: tuple[int, int] | None = None,
) -> HardwareTier:
    """Map a `Backend` enum value (as string) + optional SM pair to a tier.

    CUDA splits on compute capability: `SM >= 8.0` is the first-class
    tier (`cuda-sm80+`); older cards fall into `cuda-sm<80`.

    Callers that don't have capabilities on hand can pass `None` and
    fall back to `"cpu"` — a safe default that will surface as a
    warn-level hardware_tier mismatch when the real host is resampled.
    """
    if backend == "cuda":
        if sm is None:
            # Unknown SM — treat as legacy tier so the lock leans safe.
            return "cuda-sm<80"
        return "cuda-sm80+" if sm[0] >= 8 else "cuda-sm<80"
    if backend is None:
        return "cpu"
    tier = _BACKEND_TO_TIER.get(backend)
    if tier is None:
        return "cpu"
    return tier


def build_lock(
    *,
    dlm_id: str,
    dlm_sha256: str,
    base_model_revision: str,
    hardware_tier: HardwareTier,
    seed: int,
    determinism_class: DeterminismClass,
    run_id: int,
    pinned_versions: dict[str, str] | None = None,
    cuda_version: str | None = None,
    rocm_version: str | None = None,
    determinism_flags: dict[str, object] | None = None,
    license_acceptance: LicenseAcceptance | None = None,
    base_model_sha256: str | None = None,
) -> DlmLock:
    """Assemble a `DlmLock` with defaults for omitted fields."""
    return DlmLock(
        created_at=datetime.now(UTC),
        dlm_id=dlm_id,
        dlm_sha256=dlm_sha256,
        base_model_revision=base_model_revision,
        base_model_sha256=base_model_sha256,
        pinned_versions=pinned_versions or {},
        cuda_version=cuda_version,
        rocm_version=rocm_version,
        hardware_tier=hardware_tier,
        seed=seed,
        determinism_flags=determinism_flags or {},
        determinism_class=determinism_class,
        license_acceptance=license_acceptance,
        last_run_id=run_id,
    )
