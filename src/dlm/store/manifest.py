"""Per-store `manifest.json` — the running narrative of a training store.

The `Manifest` model owns:

- Identity: `dlm_id`, `created_at`, `updated_at`, `schema_version`.
- Base model resolution: key, HF revision, sha256.
- Adapter state: current version number and its checksums.
- Training runs and exports (typed lists).
- Content-delta tracking: `content_hashes` section_id → last-trained hash.
- Pinned versions of every tool in the stack (mirrored into `dlm.lock`).

`save_manifest` / `load_manifest` are the only supported I/O paths:
writes go to a tmp sibling then `os.replace` atomically onto the final
filename. Round-trip is **byte-identical** (sorted keys, ISO 8601
datetimes, trailing LF).

Manifest migration is intentionally separate; lower `schema_version`
values are promoted through the migration layer before normal load.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dlm.base_models.license import LicenseAcceptance
from dlm.store.errors import ManifestCorruptError, ManifestVersionError

CURRENT_MANIFEST_SCHEMA_VERSION: Final[int] = 1


# --- nested types -------------------------------------------------------------


class TrainingRunSummary(BaseModel):
    """One row in `manifest.training_runs`.

    Stored here so the manifest model validates end-to-end.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: int = Field(..., ge=1)
    started_at: datetime
    ended_at: datetime | None = None
    adapter_version: int = Field(..., ge=1)
    base_model_revision: str | None = None
    seed: int
    steps: int = Field(0, ge=0)
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    status: Literal["running", "completed", "failed", "cancelled"] = "completed"
    pinned_versions: dict[str, str] = Field(default_factory=dict)
    # Relative path (from store root) to the `logs/train-*.summary.json`
    # file written alongside the adapter. `dlm show`
    # reads this to surface "how did the last run go?" without globbing.
    # `None` for older recorded runs.
    summary_path: str | None = None
    # The named adapter this run trained (multi-adapter documents).
    # `None` for flat single-adapter runs. Without this
    # field, multi-adapter stores can't attribute a TrainingRunSummary
    # to its adapter.
    adapter_name: str | None = None
    # Per-`(tag_key, tag_value)` counts of the rows eligible for
    # tag-weighted expansion, recorded pre-expansion. `None` when no
    # directive-sourced `.dlm/training.yaml` declared a `weights`
    # block — callers can distinguish "weights inactive" from "weights
    # active with zero matching rows" (empty dict).
    weight_distribution: dict[str, dict[str, int]] | None = None


class ExportSummary(BaseModel):
    """One row in `manifest.exports`.

    Stored here for the same reason as `TrainingRunSummary`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    exported_at: datetime
    target: str = "ollama"
    quant: str
    merged: bool = False
    ollama_name: str | None = None
    ollama_version: str | None = None
    llama_cpp_tag: str | None = None
    base_gguf_sha256: str | None = None
    adapter_gguf_sha256: str | None = None
    smoke_output_first_line: str | None = None
    # The named adapter this export was built from. `None` for
    # flat/single-adapter documents and for weighted-merge exports
    # (which combine multiple adapters — see `adapter_mix`).
    adapter_name: str | None = None
    # For weighted-merge exports, the list of (name, weight) pairs
    # that produced the composite adapter. `None` for single-adapter
    # exports.
    adapter_mix: list[tuple[str, float]] | None = None


# --- Manifest itself ----------------------------------------------------------


def _utcnow() -> datetime:
    # Timezone-aware UTC (`datetime.utcnow()` is deprecated
    # in Py3.12). We strip tzinfo so the serialized JSON stays the same
    # naive-ISO-8601 form every downstream consumer already expects.
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def _empty_dict() -> dict[str, str]:
    return {}


def _empty_int_dict() -> dict[str, int]:
    return {}


def _empty_runs() -> list[TrainingRunSummary]:
    return []


def _empty_exports() -> list[ExportSummary]:
    return []


class Manifest(BaseModel):
    """Per-store narrative + content-delta index + version pins.

    Frozen — updates go through `touch()` /
    `model_copy()` to produce a new instance; mutation would bypass the
    atomic write contract anyway.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dlm_id: str = Field(..., min_length=1)
    schema_version: int = CURRENT_MANIFEST_SCHEMA_VERSION
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    base_model: str = Field(..., min_length=1)
    base_model_revision: str | None = None
    base_model_sha256: str | None = None

    adapter_version: int = Field(0, ge=0)
    adapter_sha256: str | None = None
    adapter_config_sha256: str | None = None
    # Per-adapter version attribution for multi-adapter stores
    # only. For flat stores this stays empty; for multi-adapter
    # stores it holds `{name: latest_version}` so `dlm show` can answer
    # "which adapter is at which version?" without scanning the disk.
    adapter_versions: dict[str, int] = Field(default_factory=_empty_int_dict)

    training_runs: list[TrainingRunSummary] = Field(default_factory=_empty_runs)
    exports: list[ExportSummary] = Field(default_factory=_empty_exports)

    content_hashes: dict[str, str] = Field(default_factory=_empty_dict)
    pinned_versions: dict[str, str] = Field(default_factory=_empty_dict)

    # Source `.dlm` file we were last associated with. Used for orphan
    # detection; updated by CLI operations.
    source_path: Path | None = None

    # License acceptance fingerprint. `None` on
    # non-gated bases or older stores. The per-store
    # `dlm.lock` mirrors this; divergence triggers a
    # re-check on the next `dlm train`.
    license_acceptance: LicenseAcceptance | None = None


# --- I/O ---------------------------------------------------------------------


def save_manifest(path: Path, manifest: Manifest) -> None:
    """Write `manifest` to `path` atomically.

    Serializes to deterministic JSON (sorted keys, 2-space indent,
    trailing LF) and commits via `dlm.io.atomic.write_text`.

    Round-trip is byte-identical: `save(load(save(m))) == save(m)`.
    """
    from dlm.io.atomic import write_text as _atomic_write_text

    _atomic_write_text(path, _canonical_json(manifest))


def load_manifest(path: Path) -> Manifest:
    """Read and validate `manifest.json`.

    Raises `ManifestCorruptError` with the decoded reason when parsing
    fails (invalid JSON, schema violation, unknown fields). On
    `schema_version` mismatch, the current implementation also raises —
    the migration layer replaces that refusal on older versions.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestCorruptError(path, f"read failed: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestCorruptError(path, f"invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ManifestCorruptError(path, f"top-level must be object, got {type(data).__name__}")

    version = data.get("schema_version", 1)
    if version != CURRENT_MANIFEST_SCHEMA_VERSION:
        raise ManifestVersionError(path, version, CURRENT_MANIFEST_SCHEMA_VERSION)

    try:
        return Manifest.model_validate(data)
    except ValidationError as exc:
        raise ManifestCorruptError(path, f"schema violation: {exc}") from exc


# --- internals ---------------------------------------------------------------


def _canonical_json(manifest: Manifest) -> str:
    """Deterministic JSON with sorted keys, 2-space indent, trailing LF."""
    data = manifest.model_dump(mode="json")
    # `model_dump(mode="json")` already coerces datetimes to ISO strings
    # and Paths to strings; `sort_keys=True` handles the rest.
    rendered = json.dumps(
        data,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        separators=(",", ": "),
    )
    return rendered + "\n"


# Re-export for callers that want the canonical form without touching disk.
def to_canonical_json(manifest: Manifest) -> str:
    return _canonical_json(manifest)


# Convenience: bump `updated_at` before a save.
def touch(manifest: Manifest) -> Manifest:
    """Return a copy of `manifest` with `updated_at` set to now."""
    payload: dict[str, Any] = manifest.model_dump()
    payload["updated_at"] = _utcnow()
    return Manifest.model_validate(payload)
