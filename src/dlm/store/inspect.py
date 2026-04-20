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
class NamedAdapterState:
    """Per-named-adapter snapshot for multi-adapter stores (audit-07 M2)."""

    name: str
    has_current: bool
    latest_version: int  # highest vNNNN directory on disk; 0 if none


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
    # Named multi-adapter state (audit-07 M2). Empty for flat stores;
    # populated from `adapter/<name>/versions/...` subdirectories on
    # multi-adapter stores. Used by `dlm show` to surface per-adapter
    # version pointers that the flat `adapter_version` field can't.
    named_adapters: list[NamedAdapterState] = field(default_factory=list)


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

    named_adapters = _discover_named_adapters(store)

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
        named_adapters=named_adapters,
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


_VERSION_DIR_PREFIX = "v"
_FLAT_RESERVED = frozenset({"versions", "current.txt"})


def _discover_named_adapters(store: StorePath) -> list[NamedAdapterState]:
    """Scan `adapter/` for subdirs that look like named-adapter layouts.

    Excludes the flat-layout entries (`versions/`, `current.txt`). A
    named-adapter dir is any other directory under `adapter/` that
    contains a `versions/` subdirectory — the Sprint 20a layout
    signature.
    """
    if not store.adapter.exists():
        return []
    out: list[NamedAdapterState] = []
    for entry in sorted(store.adapter.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        if entry.name in _FLAT_RESERVED:
            continue
        versions_dir = entry / "versions"
        if not versions_dir.is_dir():
            continue
        latest = _max_version(versions_dir)
        try:
            current = store.resolve_current_adapter_for(entry.name) is not None
        except (ValueError, OSError):
            current = False
        out.append(
            NamedAdapterState(
                name=entry.name,
                has_current=current,
                latest_version=latest,
            )
        )
    return out


def _max_version(versions_dir: Path) -> int:
    """Return the highest `vNNNN` directory number, or 0 if none."""
    highest = 0
    for child in versions_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith(_VERSION_DIR_PREFIX):
            continue
        try:
            n = int(name[len(_VERSION_DIR_PREFIX):])
        except ValueError:
            continue
        highest = max(highest, n)
    return highest


__all__ = [
    "ManifestCorruptError",
    "NamedAdapterState",
    "StoreInspection",
    "inspect_store",
]
