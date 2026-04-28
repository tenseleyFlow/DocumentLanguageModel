"""`export_manifest.json` schema + atomic I/O.

One file per `exports/<quant>/` directory. Records:

- export target (`ollama`, `vllm`, `llama-server`, `mlx-serve`)
- quant level
- checksums of emitted GGUF artifacts
- pinned llama.cpp tag (so a future upstream bump can detect drift)
- created_at + created_by (dlm version)
- path to the generated Modelfile
- `ollama_name` the user registered with (if provided)

Frozen + `extra="forbid"` per the project's strict-schema norm. Byte-
identical JSON round-trip via `dlm.io.atomic.write_text`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from dlm.export.errors import ExportManifestError
from dlm.io.atomic import write_text

EXPORT_MANIFEST_FILENAME = "export_manifest.json"


class ExportArtifact(BaseModel):
    """One emitted file: path relative to the export dir + sha256."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(..., min_length=1)
    sha256: str = Field(..., min_length=64, max_length=64)
    size_bytes: int = Field(..., ge=0)


class ExportManifest(BaseModel):
    """Self-describing record of one export run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str = Field("ollama", min_length=1)
    quant: str = Field(..., min_length=1)
    merged: bool = False
    dequantized: bool = False
    ollama_name: str | None = None
    created_at: datetime
    created_by: str = Field(..., description="dlm version that wrote this manifest.")
    llama_cpp_tag: str | None = None
    base_model_hf_id: str
    base_model_revision: str
    adapter_version: int = Field(..., ge=1)
    artifacts: list[ExportArtifact] = Field(default_factory=list)


def compute_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of `path`. Reused by the runner for every artifact."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_artifact(export_dir: Path, artifact_path: Path) -> ExportArtifact:
    """Assemble an `ExportArtifact` entry from a path on disk."""
    stat = artifact_path.stat()
    relative = artifact_path.resolve().relative_to(export_dir.resolve())
    return ExportArtifact(
        path=str(relative),
        sha256=compute_sha256(artifact_path),
        size_bytes=stat.st_size,
    )


def save_export_manifest(export_dir: Path, manifest: ExportManifest) -> Path:
    """Atomic JSON write to `<export_dir>/export_manifest.json`."""
    path = export_dir / EXPORT_MANIFEST_FILENAME
    payload = manifest.model_dump(mode="json")
    blob = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    write_text(path, blob)
    return path


def load_export_manifest(export_dir: Path) -> ExportManifest:
    """Read + validate the JSON manifest at `<export_dir>/export_manifest.json`."""
    path = export_dir / EXPORT_MANIFEST_FILENAME
    if not path.exists():
        raise ExportManifestError(f"missing {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportManifestError(f"cannot parse {path}: {exc}") from exc
    try:
        return ExportManifest.model_validate(data)
    except Exception as exc:  # pydantic ValidationError, etc.
        raise ExportManifestError(f"{path} has invalid shape: {exc}") from exc


def utc_now() -> datetime:
    """Tz-naive UTC with microseconds zeroed — matches the project convention."""
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)
