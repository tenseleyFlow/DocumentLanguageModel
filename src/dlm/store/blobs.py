"""Content-addressed blob store (Sprint 35 v1).

Stores binary payloads (image files for now, audio later) keyed by
sha256 of the raw bytes. The store lives under each `.dlm` store's
root at `blobs/<sha-prefix>/<sha>.<ext>` with a 256-way prefix fan-
out (first two hex chars of the sha).

The API is tight on purpose:

- `put(path) -> BlobHandle`: reads the file, hashes it, copies into
  the store, returns `(sha, ext, size)`. Idempotent — the same bytes
  re-ingested return the same handle without rewriting. Atomic write
  via `dlm.io.atomic.write_bytes`.
- `get(sha) -> Path`: resolves a sha to its on-disk path, raises
  `BlobMissingError` on miss. Callers receive a read-only path.
- `gc(live_shas) -> Iterator[BlobHandle]`: yields (and deletes)
  blobs not in the live set. Replay-corpus callers union their
  live refs with the current section set before calling.

Extensions: the store tracks a small set of known binary extensions
(`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`). Unknown
extensions fall back to `.bin` so the on-disk filename always has
a suffix — some downstream consumers (HF processors) sniff by name.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dlm.io.atomic import write_bytes
from dlm.store.errors import StoreError

_BLOB_SHA_PREFIX_LEN: Final[int] = 2
_KNOWN_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
)
_FALLBACK_EXT: Final[str] = ".bin"
_HASH_CHUNK: Final[int] = 65_536


class BlobError(StoreError):
    """Base class for blob-store errors."""


class BlobMissingError(BlobError):
    """Requested sha is not present in the blob store."""


class BlobCorruptError(BlobError):
    """On-disk blob bytes do not hash to the recorded sha."""


@dataclass(frozen=True)
class BlobHandle:
    """Identity record for a stored blob.

    `sha` is the hex-encoded sha256 of the raw bytes. `ext` is the
    stored file suffix (normalized to lowercase, `.bin` fallback for
    unknown types). `size` is the blob's byte length.
    """

    sha: str
    ext: str
    size: int


class BlobStore:
    """Lazy-initialized blob store rooted at a `.dlm` store's `blobs/` dir.

    Constructing a `BlobStore` does *not* create the directory —
    callers who never write don't trigger disk activity. The first
    `put` creates the root + prefix bucket lazily.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def put(self, path: Path) -> BlobHandle:
        """Ingest `path`'s bytes into the store; return the handle.

        Idempotent: if the sha already exists on disk, return the
        existing handle without re-copying. The caller's path is
        only used as a byte source; the store does not link to it.
        """
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        ext = _normalize_ext(path.suffix)
        dst = self._path_for(sha, ext)
        if dst.exists():
            return BlobHandle(sha=sha, ext=ext, size=len(data))
        dst.parent.mkdir(parents=True, exist_ok=True)
        write_bytes(dst, data)
        return BlobHandle(sha=sha, ext=ext, size=len(data))

    def put_bytes(self, data: bytes, *, ext: str) -> BlobHandle:
        """Ingest raw `data` with an explicit extension.

        Useful for synthetic blobs and tests — `put(path)` is the
        user-facing entry point. `ext` is normalized the same way.
        """
        sha = hashlib.sha256(data).hexdigest()
        normalized_ext = _normalize_ext(ext)
        dst = self._path_for(sha, normalized_ext)
        if dst.exists():
            return BlobHandle(sha=sha, ext=normalized_ext, size=len(data))
        dst.parent.mkdir(parents=True, exist_ok=True)
        write_bytes(dst, data)
        return BlobHandle(sha=sha, ext=normalized_ext, size=len(data))

    def get(self, sha: str) -> Path:
        """Return the on-disk path for `sha`, or raise `BlobMissingError`.

        Does *not* verify the bytes against `sha` — callers trust the
        store's filename invariant. Use `verify(sha)` when paranoia is
        justified (pack verification, audit trails).
        """
        _validate_sha(sha)
        match = self._find_path(sha)
        if match is None:
            raise BlobMissingError(f"blob {sha} not found under {self._root}")
        return match

    def exists(self, sha: str) -> bool:
        _validate_sha(sha)
        return self._find_path(sha) is not None

    def verify(self, sha: str) -> None:
        """Rehash the on-disk blob and raise `BlobCorruptError` on mismatch."""
        path = self.get(sha)
        actual = _hash_file(path)
        if actual != sha:
            raise BlobCorruptError(f"blob {sha} at {path} hashes to {actual}")

    def iter_all(self) -> Iterator[BlobHandle]:
        """Yield every blob currently on disk in deterministic order."""
        if not self._root.exists():
            return
        for bucket in sorted(self._root.iterdir()):
            if not bucket.is_dir():
                continue
            for child in sorted(bucket.iterdir()):
                if not child.is_file():
                    continue
                yield _handle_from_path(child)

    def gc(self, live_shas: set[str]) -> Iterator[BlobHandle]:
        """Delete every blob whose sha is not in `live_shas`; yield the deleted handles.

        Caller passes the union of all sections referencing blobs
        (current parse + replay-addressable sections). Safe to call
        repeatedly; noop on a fresh store.
        """
        for handle in list(self.iter_all()):
            if handle.sha in live_shas:
                continue
            path = self._path_for(handle.sha, handle.ext)
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            yield handle

    def clear(self) -> None:
        """Delete the entire blob tree. Test + `dlm unpack --force` only."""
        if self._root.exists():
            shutil.rmtree(self._root)

    # --- internals ----------------------------------------------------------

    def _path_for(self, sha: str, ext: str) -> Path:
        prefix = sha[:_BLOB_SHA_PREFIX_LEN]
        return self._root / prefix / f"{sha}{ext}"

    def _find_path(self, sha: str) -> Path | None:
        prefix = sha[:_BLOB_SHA_PREFIX_LEN]
        bucket = self._root / prefix
        if not bucket.is_dir():
            return None
        for child in bucket.iterdir():
            if child.is_file() and child.stem == sha:
                return child
        return None


def _normalize_ext(raw: str) -> str:
    """Lowercase + fallback unknown suffixes to `.bin`.

    Accepts either `"png"` or `".png"` — both normalize to `.png`.
    Empty input returns the fallback.
    """
    if not raw:
        return _FALLBACK_EXT
    candidate = raw if raw.startswith(".") else f".{raw}"
    lowered = candidate.lower()
    if lowered in _KNOWN_EXTENSIONS:
        return lowered
    return _FALLBACK_EXT


def _validate_sha(sha: str) -> None:
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError(f"blob sha must be 64-char lowercase hex, got {sha!r}")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _handle_from_path(path: Path) -> BlobHandle:
    sha = path.stem
    _validate_sha(sha)
    ext = _normalize_ext(path.suffix)
    size = path.stat().st_size
    return BlobHandle(sha=sha, ext=ext, size=size)
