"""Content-addressed on-disk store for a `.dlm` file.

The store lives at `{$DLM_HOME}/store/<dlm_id>/` and holds adapter
weights, optimizer state, replay corpus, and exports. This module owns
path resolution, the manifest schema + atomic I/O, the exclusive file
lock, and read-only inspection for `dlm show`.

Sprint 04 lands the skeleton and contracts. Sprints 08/09/11/12/12b
populate the subdirs and extend the manifest.
"""

from __future__ import annotations

from dlm.store.blobs import BlobCorruptError, BlobHandle, BlobMissingError, BlobStore
from dlm.store.errors import (
    LockHeldError,
    ManifestCorruptError,
    ManifestVersionError,
    OrphanedStoreError,
    StaleLockError,
    StoreError,
    UnknownStoreError,
)
from dlm.store.inspect import StoreInspection, inspect_store
from dlm.store.lock import LockInfo, break_lock, exclusive
from dlm.store.manifest import (
    CURRENT_MANIFEST_SCHEMA_VERSION,
    ExportSummary,
    Manifest,
    TrainingRunSummary,
    load_manifest,
    save_manifest,
    to_canonical_json,
    touch,
)
from dlm.store.paths import StorePath, dlm_home, ensure_home, for_dlm

__all__ = [
    "CURRENT_MANIFEST_SCHEMA_VERSION",
    "BlobCorruptError",
    "BlobHandle",
    "BlobMissingError",
    "BlobStore",
    "ExportSummary",
    "LockHeldError",
    "LockInfo",
    "Manifest",
    "ManifestCorruptError",
    "ManifestVersionError",
    "OrphanedStoreError",
    "StaleLockError",
    "StoreError",
    "StoreInspection",
    "StorePath",
    "TrainingRunSummary",
    "UnknownStoreError",
    "break_lock",
    "dlm_home",
    "ensure_home",
    "exclusive",
    "for_dlm",
    "inspect_store",
    "load_manifest",
    "save_manifest",
    "to_canonical_json",
    "touch",
]
