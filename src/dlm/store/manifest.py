"""Per-store `manifest.json` — the running narrative of a training store.

The `Manifest` model owns:

- Identity: `dlm_id`, `created_at`, `updated_at`, `schema_version`.
- Base model resolution: key, HF revision, sha256.
- Adapter state: current version number and its checksums.
- Training runs and exports (typed lists, populated by Sprints 09 and 12).
- Content-delta tracking: `content_hashes` section_id → last-trained hash.
- Pinned versions of every tool in the stack (mirrored into `dlm.lock`).

`save_manifest` / `load_manifest` are the only supported I/O paths:
writes go to a tmp sibling then `os.replace` atomically onto the final
filename. Round-trip is **byte-identical** (sorted keys, ISO 8601
datetimes, trailing LF).

Sprint 12b owns the schema migration framework; when it lands,
`load_manifest` will dispatch through the migrator on a lower
`schema_version`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dlm.store.errors import ManifestCorruptError

CURRENT_MANIFEST_SCHEMA_VERSION: Final[int] = 1


# --- nested types -------------------------------------------------------------


class TrainingRunSummary(BaseModel):
    """One row in `manifest.training_runs`.

    Owned by Sprint 09 (which writes instances); Sprint 04 ships the
    schema so the Manifest model validates end-to-end today.
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


class ExportSummary(BaseModel):
    """One row in `manifest.exports`.

    Owned by Sprint 12 (which writes instances); same rationale as
    TrainingRunSummary — shipping the type here unblocks validation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    exported_at: datetime
    quant: str
    merged: bool = False
    ollama_name: str | None = None
    ollama_version: str | None = None
    llama_cpp_tag: str | None = None
    base_gguf_sha256: str | None = None
    adapter_gguf_sha256: str | None = None
    smoke_output_first_line: str | None = None


# --- Manifest itself ----------------------------------------------------------


def _utcnow() -> datetime:
    # Naive UTC — JSON round-trips cleanly without tzinfo bookkeeping.
    return datetime.utcnow().replace(microsecond=0)


def _empty_dict() -> dict[str, str]:
    return {}


def _empty_runs() -> list[TrainingRunSummary]:
    return []


def _empty_exports() -> list[ExportSummary]:
    return []


class Manifest(BaseModel):
    """Per-store narrative + content-delta index + version pins."""

    model_config = ConfigDict(extra="forbid")

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

    training_runs: list[TrainingRunSummary] = Field(default_factory=_empty_runs)
    exports: list[ExportSummary] = Field(default_factory=_empty_exports)

    content_hashes: dict[str, str] = Field(default_factory=_empty_dict)
    pinned_versions: dict[str, str] = Field(default_factory=_empty_dict)

    # Source `.dlm` file we were last associated with. Used for orphan
    # detection (Sprint 04 `inspect`); updated by CLI ops (Sprint 13).
    source_path: Path | None = None


# --- I/O ---------------------------------------------------------------------


def save_manifest(path: Path, manifest: Manifest) -> None:
    """Write `manifest` to `path` atomically.

    1. Serialize to deterministic JSON (sorted keys, 2-space indent,
       trailing LF).
    2. Write to a tmp sibling file.
    3. `os.replace` onto the final name.

    Round-trip is byte-identical: `save(load(save(m))) == save(m)`.
    """
    payload = _canonical_json(manifest)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def load_manifest(path: Path) -> Manifest:
    """Read and validate `manifest.json`.

    Raises `ManifestCorruptError` with the decoded reason when parsing
    fails (invalid JSON, schema violation, unknown fields). On
    `schema_version` mismatch, the current implementation also raises —
    Sprint 12b's migrator will replace this behavior with a dispatch.
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
        raise ManifestCorruptError(
            path,
            f"schema_version {version} requires migration to "
            f"{CURRENT_MANIFEST_SCHEMA_VERSION} (Sprint 12b owns the framework)",
        )

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
