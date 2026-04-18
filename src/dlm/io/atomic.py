"""Atomic filesystem writes.

Single entry point for "write to a tmp sibling then `os.replace` onto the
final name." Three Phase-0 modules (`dlm.io.text`, `dlm.store.manifest`,
`dlm.store.paths`) independently grew this pattern; consolidating here
gives us one place to add `fsync` / directory-sync semantics later.

The tmp file carries the writer's PID so concurrent writers don't stomp
each other's scratch files mid-write. After a crash, stale `.tmp.<pid>`
files are harmless: they sit next to the real file and are swept up by
`cleanup_stale_tmp_files` from within sprints that notice them (e.g.,
Sprint 04's store load path).
"""

from __future__ import annotations

import os
from pathlib import Path


def write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace `path` with `data`.

    Writes to `<path>.tmp.<pid>`, then `os.replace()` to the final name.
    Atomic on POSIX and on Windows NTFS. Parent directory must exist.
    """
    tmp = _tmp_path(path)
    tmp.write_bytes(data)
    tmp.replace(path)


def write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace `path` with `text` encoded via `encoding`.

    Defaults to UTF-8 (the project-wide text contract — see `dlm.io.text`).
    """
    write_bytes(path, text.encode(encoding))


def cleanup_stale_tmp_files(directory: Path) -> list[Path]:
    """Remove `*.tmp.<pid>` files in `directory` whose PID is no longer alive.

    Returns the list of files actually removed. Safe to call on a missing
    directory (returns empty). Never removes the final target or files
    belonging to live PIDs.
    """
    if not directory.is_dir():
        return []
    removed: list[Path] = []
    for child in directory.iterdir():
        if not child.is_file():
            continue
        pid = _tmp_pid(child)
        if pid is None or _is_alive(pid):
            continue
        try:
            child.unlink()
        except FileNotFoundError:
            continue
        removed.append(child)
    return removed


# --- internals ---------------------------------------------------------------


def _tmp_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + f".tmp.{os.getpid()}")


def _tmp_pid(path: Path) -> int | None:
    """Return the PID embedded in a `<name>.tmp.<pid>` filename, or None."""
    name = path.name
    marker = ".tmp."
    idx = name.rfind(marker)
    if idx == -1:
        return None
    try:
        return int(name[idx + len(marker) :])
    except ValueError:
        return None


def _is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
