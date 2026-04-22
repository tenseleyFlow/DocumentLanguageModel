"""Atomic filesystem writes.

Single entry point for "write to a tmp sibling then `os.replace` onto the
final name." Three early modules (`dlm.io.text`, `dlm.store.manifest`,
`dlm.store.paths`) independently grew this pattern; consolidating here
gives us one place to add `fsync` / directory-sync semantics later.

Each tmp file carries the writer's PID plus a random 8-hex-char nonce so
concurrent writers don't stomp each other's scratch files — and so PID
reuse (after a parent process dies and the kernel recycles the number)
can't make a stale tmp match a live peer. After a crash, stale tmps are
harmless: they sit next to the real file and are swept up by
`cleanup_stale_tmp_files` from the store load path.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace `path` with `data`.

    Writes to `<path>.tmp.<pid>.<nonce>`, then `os.replace()` to the
    final name. Atomic on POSIX and on Windows NTFS. Parent directory
    must exist.
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
    """Remove tmp-suffix files in `directory` whose PID is no longer alive.

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

# `<suffix>.tmp.<pid>.<nonce>` where nonce is 8 hex chars. The nonce
# makes PID reuse harmless: a recycled PID on a stale tmp can't collide
# with a live peer because the nonce differs.
_TMP_RE = re.compile(r"\.tmp\.(?P<pid>\d+)\.(?P<nonce>[0-9a-f]{8})$")


def _tmp_path(path: Path) -> Path:
    nonce = os.urandom(4).hex()
    return path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{nonce}")


def _tmp_pid(path: Path) -> int | None:
    """Return the PID embedded in a tmp-suffixed filename, or None.

    Accepts both the nonce-suffixed shape (`<name>.tmp.<pid>.<hex8>`)
    and the legacy nonce-less shape (`<name>.tmp.<pid>`) so sweeps on a
    store that spans a pre- and post-upgrade writer still clean up
    correctly.
    """
    name = path.name
    m = _TMP_RE.search(name)
    if m is not None:
        try:
            return int(m.group("pid"))
        except ValueError:
            return None
    # Legacy fallback: `<name>.tmp.<pid>` with no nonce.
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
