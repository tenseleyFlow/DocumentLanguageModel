"""Path confinement, binary-file detection, and size-cap enforcement.

Three primitives that the expansion loop composes:

- `confine_path(path, root, strict)` — resolves symlinks and, under
  strict policy, verifies `resolved.is_relative_to(root)`. Strict
  mode raises `DirectivePolicyError`; permissive logs a warning for
  symlink escapes and proceeds.
- `is_probably_binary(data)` — NUL-byte scan of the first KiB. The
  standard heuristic (git, grep). Catches images, archives, compiled
  objects; misses text-encoded binaries (base64 blobs) but those are
  legitimately training material.
- `enumerate_with_caps(root, include, exclude, max_files,
  max_bytes_per_file)` — deterministic lexicographic walk with
  include/exclude glob filtering + size/count caps.

All three stay pure Python (no third-party deps) so test doubles are
easy to set up in `tmp_path`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from dlm.directives.errors import DirectivePolicyError

__all__ = [
    "confine_path",
    "enumerate_matching_files",
    "is_probably_binary",
]

_LOG = logging.getLogger(__name__)

_BINARY_SNIFF_BYTES = 1024


def _compile_glob(pattern: str) -> re.Pattern[str]:
    """Translate a `**`-aware glob to a regex matching POSIX-style paths.

    Rules:
    - `**` matches any number of path segments (including zero).
    - `*` matches any run of non-`/` characters.
    - `?` matches a single non-`/` character.
    - Other characters are literal (regex-escaped).

    Trailing-`/**` is treated as "anything beneath this prefix" —
    `tests/**` matches `tests/a`, `tests/a/b`, etc.
    """
    i = 0
    n = len(pattern)
    out: list[str] = ["^"]
    while i < n:
        c = pattern[i]
        if c == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                out.append(".*")
                i += 2
                # consume a trailing `/` after `**` so `tests/**/x` matches
                # both `tests/x` and `tests/a/b/x`
                if i < n and pattern[i] == "/":
                    i += 1
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    out.append("$")
    return re.compile("".join(out))


def confine_path(path: Path, root: Path, *, strict: bool) -> Path:
    """Resolve `path` and, under strict policy, verify containment under
    `root`. Returns the resolved absolute path.

    The resolve uses `strict=False` on the Path so callers get a
    meaningful `DirectivePathError` from upstream rather than a
    `FileNotFoundError` here; the caller validates existence
    separately via `path.exists()`.

    Permissive policy still *resolves* the path (to normalize ~ and
    symlinks) but doesn't enforce containment. Symlink escapes under
    permissive log one WARN so operators see the escape in training
    logs without failing the run.
    """
    resolved = path.expanduser().resolve()
    root_resolved = root.expanduser().resolve()
    if strict:
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise DirectivePolicyError(resolved, root_resolved) from exc
    else:
        # Permissive: only log if the path resolves outside `root` AND
        # the original path started *inside* it (i.e., a symlink escape).
        # A plain `~/elsewhere` path isn't an escape, just an external
        # source — that's the whole point of permissive mode.
        try:
            path.relative_to(root)
        except ValueError:
            pass  # not anchored at root → not an escape
        else:
            try:
                resolved.relative_to(root_resolved)
            except ValueError:
                _LOG.warning(
                    "directive: symlink at %s escapes %s (permissive: proceeding)",
                    path,
                    root,
                )
    return resolved


def is_probably_binary(data: bytes, *, sample: int = _BINARY_SNIFF_BYTES) -> bool:
    """Return True if `data[:sample]` contains a NUL byte.

    Fast and conservative: UTF-8 text never contains NUL outside
    explicit escapes, and every common binary format does.
    """
    return b"\x00" in data[:sample]


def enumerate_matching_files(
    root: Path,
    *,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> Iterator[Path]:
    """Yield files under `root` matching include patterns, skipping
    excluded ones. Deterministic: lexicographic sort.

    If `root` is a file, yield it iff it matches the filters (path
    patterns are checked against the name component alone). If `root`
    is a directory, walk it and match paths relative to `root`.

    Size and count caps are enforced by the caller so skip counts
    can be recorded in provenance.
    """
    if root.is_file():
        if _matches_filters(root.name, include, exclude):
            yield root
        return

    if not root.is_dir():
        return

    candidates = sorted(p for p in root.rglob("*") if p.is_file())
    for candidate in candidates:
        rel = candidate.relative_to(root).as_posix()
        if _matches_filters(rel, include, exclude):
            yield candidate


def _matches_filters(rel_path: str, include: Iterable[str], exclude: Iterable[str]) -> bool:
    """Match rel_path against include (any) and exclude (none)."""
    if any(_compile_glob(pat).fullmatch(rel_path) for pat in exclude):
        return False
    return any(_compile_glob(pat).fullmatch(rel_path) for pat in include)
