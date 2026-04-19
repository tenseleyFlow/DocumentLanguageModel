"""In-pack directory layout + required entries (Sprint 14).

All filenames + directory prefixes live here so the packer, unpacker,
and integrity module share one source of truth. Changes here are
breaking to the pack format and require a `pack_format_version` bump.
"""

from __future__ import annotations

from typing import Final

# Top-level entries inside the tarball.
HEADER_FILENAME: Final[str] = "PACK_HEADER.json"
MANIFEST_FILENAME: Final[str] = "manifest.json"
SHA256_FILENAME: Final[str] = "CHECKSUMS.sha256"

# Subtree prefixes.
DLM_DIR: Final[str] = "dlm"
STORE_DIR: Final[str] = "store"
# Optional subtrees — packer includes only with --include-exports /
# --include-base. The unpacker uses these names to detect presence.
STORE_EXPORTS_DIR: Final[str] = "store/exports"
STORE_CACHE_DIR: Final[str] = "store/cache"
STORE_LOGS_DIR: Final[str] = "store/logs"

# Required entries any valid v1 pack must carry. Missing ⇒
# `PackLayoutError` at unpack. `dlm/` and `store/manifest.json` are
# required even in minimal packs so the restored `.dlm` is complete.
REQUIRED_ENTRIES: Final[tuple[str, ...]] = (
    HEADER_FILENAME,
    MANIFEST_FILENAME,
    SHA256_FILENAME,
    f"{STORE_DIR}/manifest.json",
)
