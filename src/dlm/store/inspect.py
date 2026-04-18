"""Read-only store introspection.

`inspect_store()` walks a store, computes summary statistics, and — when
given the path to the source `.dlm` — flags orphaned stores whose source
is missing or has the wrong `dlm_id`. The returned `StoreInspection`
dataclass is the backing for Sprint 13's `dlm show` CLI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dlm.doc.errors import DlmParseError
from dlm.doc.parser import parse_file
from dlm.io.text import DlmEncodingError
from dlm.store.errors import ManifestCorruptError
from dlm.store.manifest import ExportSummary, Manifest, load_manifest
from dlm.store.paths import StorePath

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoreInspection:
    """Summary of a store at a point in time.

    All sizes are in bytes. `orphaned` is True iff a `source_path` was
    provided (either via the manifest or as an argument) and the file at
    that path either doesn't exist or parses to a different `dlm_id`.
    """

    dlm_id: str
    path: Path
    base_model: str
    base_model_revision: str | None
    adapter_version: int
    training_runs: int
    last_trained_at: datetime | None
    exports: list[ExportSummary]
    has_adapter_current: bool
    replay_size_bytes: int
    total_size_bytes: int
    source_path: Path | None
    orphaned: bool
    # Mirrored from the manifest so the CLI can render them without a
    # second load; keep in sync with Sprint 13's output format.
    content_hashes: dict[str, str] = field(default_factory=dict)
    pinned_versions: dict[str, str] = field(default_factory=dict)


def inspect_store(store: StorePath, *, source_path: Path | None = None) -> StoreInspection:
    """Walk `store` and produce a `StoreInspection`.

    Raises `ManifestCorruptError` if the manifest is unreadable — caller
    decides whether to treat that as user-facing error (Sprint 13 will).
    """
    manifest = load_manifest(store.manifest)
    effective_source = source_path or manifest.source_path
    orphaned = _is_orphaned(effective_source, manifest.dlm_id)

    total_size = _directory_size(store.root)
    replay_size = _directory_size(store.replay) if store.replay.exists() else 0

    last_trained_at = _last_trained_at(manifest)

    return StoreInspection(
        dlm_id=manifest.dlm_id,
        path=store.root,
        base_model=manifest.base_model,
        base_model_revision=manifest.base_model_revision,
        adapter_version=manifest.adapter_version,
        training_runs=len(manifest.training_runs),
        last_trained_at=last_trained_at,
        exports=list(manifest.exports),
        has_adapter_current=store.resolve_current_adapter() is not None,
        replay_size_bytes=replay_size,
        total_size_bytes=total_size,
        source_path=effective_source,
        orphaned=orphaned,
        content_hashes=dict(manifest.content_hashes),
        pinned_versions=dict(manifest.pinned_versions),
    )


# --- helpers -----------------------------------------------------------------


def _is_orphaned(source_path: Path | None, expected_dlm_id: str) -> bool:
    """Return True iff the linked `.dlm` cannot be confirmed to match.

    We treat only expected failure modes as "can't confirm" → orphan:
    missing file, parse errors, invalid encoding, or I/O errors reading
    the file. Unexpected exceptions (e.g., a bug in the parser) are left
    to propagate so bugs surface instead of silently marking stores
    orphan (audit-03 M1).
    """
    if source_path is None:
        return False
    if not source_path.exists():
        return True
    try:
        parsed = parse_file(source_path)
    except (DlmParseError, DlmEncodingError, OSError) as exc:
        _LOG.warning(
            "orphan probe: could not parse %s: %s: %s",
            source_path,
            type(exc).__name__,
            exc,
        )
        return True
    return parsed.frontmatter.dlm_id != expected_dlm_id


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                # Transient file disappearance: ignore.
                continue
    return total


def _last_trained_at(manifest: Manifest) -> datetime | None:
    if not manifest.training_runs:
        return None
    finished = [run for run in manifest.training_runs if run.ended_at is not None]
    if not finished:
        return None
    return max(run.ended_at for run in finished)  # type: ignore[type-var]


__all__ = ["StoreInspection", "inspect_store", "ManifestCorruptError"]
