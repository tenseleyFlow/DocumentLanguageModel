"""Two-phase checkpoint commit (audit F12).

Invariant
---------

`adapter/current.txt` **never** points at a half-written version
directory. Either the old version is still authoritative, or the new
one is — never a partial state.

Lifecycle
---------

1. `allocate_next_version(store)` picks `vNNNN` where `NNNN` is one
   above the highest existing version directory (or `0001` on a fresh
   store). Creates the empty directory.
2. Caller populates the directory — `adapter.save_pretrained()` writes
   the adapter config + weights; `state_sidecar.save_state()` writes
   `training_state.pt` + sha256.
3. `fsync_dir(path)` flushes the directory entry to disk.
4. `store.set_current_adapter(path)` atomically flips the pointer via
   `os.replace` on a tmp file (already implemented in Sprint 04).

The `commit_version()` helper bundles steps 1 + 3 + 4 around a
caller-supplied writer function, so the "happy path" is one call. If
the writer raises, the pending directory is *not* made current — it's
left in place so the caller can inspect / clean up / retry.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dlm.store.paths import StorePath

# Regex-safe prefix shared with `StorePath.adapter_version`.
_VERSION_PREFIX = "v"


def allocate_next_version(
    store: StorePath, *, adapter_name: str | None = None
) -> Path:
    """Return the next empty `adapter/[<name>/]versions/vNNNN/` path.

    Creates the directory (and any missing parents). When `adapter_name`
    is provided, allocates under the named layout; otherwise uses the
    flat single-adapter layout.
    """
    existing = _existing_versions(store, adapter_name=adapter_name)
    next_n = (max(existing) if existing else 0) + 1
    if adapter_name is None:
        version_dir = store.adapter_version(next_n)
    else:
        version_dir = store.adapter_version_for(adapter_name, next_n)
    version_dir.mkdir(parents=True, exist_ok=False)
    return version_dir


def commit_version(
    store: StorePath,
    writer: Callable[[Path], None],
    *,
    adapter_name: str | None = None,
) -> Path:
    """Allocate → populate → fsync → flip pointer.

    Returns the committed version directory. On writer exception:
    - the pending directory is left on disk (not cleaned up, so the
      caller can diagnose)
    - the current pointer is NOT updated
    - the exception propagates
    """
    pending = allocate_next_version(store, adapter_name=adapter_name)
    try:
        writer(pending)
    except BaseException:
        # Leave `pending` on disk; the next allocate_next_version call
        # skips over it by bumping NNNN. Cleanup is a caller concern.
        raise

    fsync_dir(pending)
    if adapter_name is None:
        store.set_current_adapter(pending)
    else:
        store.set_current_adapter_for(adapter_name, pending)
    return pending


def fsync_dir(path: Path) -> None:
    """Flush the directory entry for `path` to disk.

    After writing the adapter files we need to ensure the directory
    metadata (new file entries) survives a power loss. `os.fsync` on
    the directory fd is the POSIX idiom. Windows doesn't allow opening
    a directory handle for fsync; on Windows we no-op (the underlying
    filesystem typically metadata-journals).
    """
    if os.name == "nt":  # pragma: no cover — macOS/Linux covered
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def list_pending_versions(
    store: StorePath, *, adapter_name: str | None = None
) -> list[Path]:
    """Return version dirs that exist on disk but aren't the current pointer.

    Used by the trainer's startup routine to detect crash-before-flip
    remnants: if the pending dir has a complete adapter + training_state
    + matching sha256, the user could in principle resume from it by
    manually flipping the pointer. We surface them rather than
    auto-deleting.
    """
    existing = _existing_versions(store, adapter_name=adapter_name)
    if adapter_name is None:
        current = store.resolve_current_adapter()
        version_for = store.adapter_version
    else:
        current = store.resolve_current_adapter_for(adapter_name)
        version_for = lambda n: store.adapter_version_for(adapter_name, n)  # noqa: E731
    current_n = _parse_version_number(current) if current is not None else None
    return [version_for(n) for n in sorted(existing) if n != current_n]


def _existing_versions(
    store: StorePath, *, adapter_name: str | None = None
) -> list[int]:
    base = (
        store.adapter_versions
        if adapter_name is None
        else store.adapter_versions_for(adapter_name)
    )
    if not base.is_dir():
        return []
    out: list[int] = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        n = _parse_version_dirname(entry.name)
        if n is not None:
            out.append(n)
    return out


def _parse_version_dirname(name: str) -> int | None:
    if not name.startswith(_VERSION_PREFIX):
        return None
    try:
        return int(name[len(_VERSION_PREFIX) :])
    except ValueError:
        return None


def _parse_version_number(path: Path) -> int | None:
    return _parse_version_dirname(path.name)
