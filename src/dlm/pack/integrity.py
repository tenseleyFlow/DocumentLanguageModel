"""Per-file sha256 integrity for `.dlm.pack` (Sprint 14).

Two pure functions:

- `write_checksums(root, exclude=None)` — walks `root`, hashes every
  regular file, writes `CHECKSUMS.sha256` (`<sha256>  <relpath>\\n` per
  line, LF-only, sorted by relpath) and returns the dict.
- `verify_checksums(root)` — reads `CHECKSUMS.sha256`, rehashes every
  entry, raises `PackIntegrityError` on the first mismatch.

`content_sha256` (a rollup over the per-file hashes) is computed
separately so `PackManifest.content_sha256` can validate quickly
without reopening files. Identical inputs always produce identical
rollups (sorted by relpath, newline-joined).

Heavy: chunks files at 1 MB to cap memory on large GGUFs. The outer
pack tar is zstd-compressed but the individual files on-disk during
pack assembly are uncompressed (the packer streams pre-compressed
blobs without re-compression — audit F18).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from dlm.pack.errors import PackIntegrityError
from dlm.pack.layout import SHA256_FILENAME

_HASH_CHUNK_BYTES: Final[int] = 1 << 20  # 1 MB


def sha256_of_file(path: Path) -> str:
    """Hex-encoded sha256 of `path`, chunked to cap memory on large files."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def write_checksums(
    root: Path, *, exclude: Iterable[str] | None = None
) -> dict[str, str]:
    """Walk `root` and write `CHECKSUMS.sha256`; return `{relpath: sha256}`.

    `exclude` is an iterable of *relpath strings* (forward slashes) to
    skip — typically the CHECKSUMS file itself plus any file generated
    after this call. Symlinks and special files are skipped silently.

    Output format: `<hex_sha256>  <relpath>\\n`, one entry per line,
    **sorted by relpath**, LF-only. Identical inputs produce
    byte-identical checksums files.
    """
    excluded = set(exclude or ())
    excluded.add(SHA256_FILENAME)

    checksums: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relpath = path.relative_to(root).as_posix()
        if relpath in excluded:
            continue
        checksums[relpath] = sha256_of_file(path)

    (root / SHA256_FILENAME).write_text(
        "".join(f"{hash_hex}  {relpath}\n" for relpath, hash_hex in checksums.items()),
        encoding="utf-8",
    )
    return checksums


def read_checksums(root: Path) -> dict[str, str]:
    """Parse `CHECKSUMS.sha256` back into `{relpath: sha256}`.

    Raises `PackIntegrityError` if the file is missing or malformed.
    """
    path = root / SHA256_FILENAME
    if not path.is_file():
        raise PackIntegrityError(
            relpath=SHA256_FILENAME,
            expected="<present>",
            actual="<missing>",
        )

    entries: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        # Each line is `<64-hex>  <relpath>` (two-space separator).
        parts = raw.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise PackIntegrityError(
                relpath=f"{SHA256_FILENAME}:{lineno}",
                expected="<64-hex>  <relpath>",
                actual=raw,
            )
        entries[parts[1]] = parts[0]
    return entries


def verify_checksums(root: Path) -> None:
    """Verify every entry in `CHECKSUMS.sha256` against its file.

    Raises `PackIntegrityError` on:
    - a missing file listed in the manifest,
    - a file whose sha256 differs from the manifest.

    Files present on disk but not in the manifest are NOT flagged here
    (that's a layout concern — `PackLayoutError` from the unpacker).
    """
    expected = read_checksums(root)
    for relpath, expected_hash in expected.items():
        path = root / relpath
        if not path.is_file():
            raise PackIntegrityError(
                relpath=relpath, expected=expected_hash, actual="<missing>"
            )
        actual_hash = sha256_of_file(path)
        if actual_hash != expected_hash:
            raise PackIntegrityError(
                relpath=relpath, expected=expected_hash, actual=actual_hash
            )


def rollup_sha256(checksums: dict[str, str]) -> str:
    """Deterministic rollup of per-file hashes for `PackManifest.content_sha256`.

    Sorts by relpath, joins as `<relpath>\\n<sha256>\\n`, hashes the
    resulting string. Two packers writing the same content produce the
    same rollup → identical `manifest.json` → byte-identical pack
    artifacts (modulo tar metadata).
    """
    hasher = hashlib.sha256()
    for relpath in sorted(checksums):
        hasher.update(f"{relpath}\n{checksums[relpath]}\n".encode("utf-8"))
    return hasher.hexdigest()
