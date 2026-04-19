"""Pinned, hashed base-model downloader.

Every base we train on flows through `download_spec()`:

1. `huggingface_hub.snapshot_download` with `revision=spec.revision`
   fetches the exact commit. HF's own layer verifies per-file against
   its ETag/hash index.
2. We read the resolved snapshot's commit SHA back (HF stores it) and
   refuse if it doesn't match what we asked for — branch races are
   real on non-pinned tooling.
3. A deterministic `sha256` over `(relative_path, file_sha256)` pairs
   produces a single digest for the manifest, so future runs can detect
   a base tampered with on disk.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from dlm.base_models.errors import GatedModelError
from dlm.base_models.schema import BaseModelSpec

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a successful `download_spec()` call."""

    spec: BaseModelSpec
    path: Path
    revision: str
    sha256: str


def download_spec(
    spec: BaseModelSpec,
    *,
    cache_dir: Path | None = None,
    local_dir: Path | None = None,
    local_files_only: bool = False,
) -> DownloadResult:
    """Fetch (or locate) the snapshot for `spec` and return a pinned reference.

    `cache_dir` overrides `HF_HOME`. `local_dir` copies the snapshot
    into a specific path (non-symlinked) — used by sprint 04's store
    when we want a per-`.dlm` cache. `local_files_only=True` refuses to
    hit the network (mirrors `HF_HUB_OFFLINE`).
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    try:
        snapshot = snapshot_download(
            repo_id=spec.hf_id,
            revision=spec.revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_dir=str(local_dir) if local_dir else None,
            local_files_only=local_files_only,
        )
    except GatedRepoError as exc:
        raise GatedModelError(spec.hf_id, spec.license_url) from exc
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(f"{spec.hf_id} not found in local cache and offline mode is on") from exc
    except RepositoryNotFoundError as exc:
        raise RuntimeError(f"HF repository not found: {spec.hf_id}") from exc

    path = Path(snapshot)
    resolved_revision = _resolve_revision(path, spec.revision)
    if resolved_revision != spec.revision:
        raise RuntimeError(
            f"revision mismatch for {spec.hf_id}: asked {spec.revision}, got {resolved_revision}",
        )

    digest = sha256_of_directory(path)
    return DownloadResult(
        spec=spec,
        path=path,
        revision=resolved_revision,
        sha256=digest,
    )


def sha256_of_directory(root: Path) -> str:
    """Deterministic content digest over every file under `root`.

    Input: `(posix-relative-path, sha256-of-contents)` pairs sorted by
    path. Output: hex digest. Stable across filesystems regardless of
    inode order and symlink resolution.
    """
    if not root.is_dir():
        raise NotADirectoryError(root)

    aggregator = hashlib.sha256()
    for child in sorted(root.rglob("*")):
        if not child.is_file():
            continue
        rel = child.relative_to(root).as_posix()
        aggregator.update(rel.encode("utf-8"))
        aggregator.update(b"\0")
        aggregator.update(_sha256_of_file(child).encode("ascii"))
        aggregator.update(b"\n")
    return aggregator.hexdigest()


# --- internals ---------------------------------------------------------------


def _resolve_revision(path: Path, expected: str) -> str:
    """Return the commit SHA HF wrote into the snapshot, falling back
    to the expected value when HF didn't emit a `.gitcommit` marker.

    `snapshot_download` uses the real SHA as the directory name under
    `snapshots/`, so resolving from the path is authoritative; we only
    fall back when `local_dir` copies files out of that structure.
    """
    # HF's canonical snapshot dir is `.../snapshots/<sha>/`. Walking up
    # until we find that name is the cheapest check.
    for ancestor in (path, *path.parents):
        if ancestor.parent.name == "snapshots":
            return ancestor.name
    # Not inside the canonical layout (e.g., `local_dir` copy). Trust
    # the expected SHA — snapshot_download would have already refused
    # if the revision didn't resolve.
    return expected


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
