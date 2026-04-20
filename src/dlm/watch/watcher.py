"""Filesystem-change watcher for `dlm train --watch`.

Wraps `watchfiles.watch` so tests can substitute an iterator stub.
The real watchfiles library is a runtime-only dep (runs a Rust
notify backend); this module imports it lazily so collection is
cheap.

### Editor write patterns

Vim writes `.swp` → swap → rename atomically; VS Code usually
overwrites in place; some editors hold an open handle and fsync.
`watchfiles` surfaces all three as `Change.added` / `Change.modified`
events on the target path. We watch the **parent directory** and
filter by filename, because an atomic rename briefly replaces the
file and watching the path directly can miss events on some
platforms.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from dlm.watch.errors import WatchSetupError


def _default_event_stream(
    path: Path,
    *,
    stop_event: Any | None = None,
) -> Iterator[set[tuple[Any, str]]]:  # pragma: no cover - real FS watcher
    """Real `watchfiles` event stream, yielding the set of changes per batch.

    Wrapped so tests can supply a stub iterator without importing
    watchfiles at all.
    """
    from watchfiles import watch

    yield from watch(str(path.parent), stop_event=stop_event)


def filter_events_for_path(
    batch: set[tuple[Any, str]],
    target: Path,
) -> bool:
    """True iff any event in `batch` concerns `target`.

    `watchfiles` yields (`Change`, str_path) tuples. We compare by
    resolved path so symlinks + different-case cwd stays robust.
    """
    try:
        resolved_target = str(target.resolve())
    except OSError:
        resolved_target = str(target)
    for _change, raw_path in batch:
        try:
            if str(Path(raw_path).resolve()) == resolved_target:
                return True
        except OSError:
            if raw_path == str(target):
                return True
    return False


def watch_for_changes(
    path: Path,
    on_change: Callable[[], None],
    *,
    stop_event: Any | None = None,
    event_stream: Callable[..., Iterator[set[tuple[Any, str]]]] | None = None,
) -> None:  # pragma: no cover - driven by slow integration + test seams
    """Block and call `on_change()` each time `path` is modified.

    Iterates `event_stream(path, stop_event=...)` until the iterator
    ends (caller signalled stop). Test callers pass a custom
    `event_stream` that yields a deterministic sequence.

    No debouncing happens here — that's `Debouncer`'s job, driven by
    the calling loop.
    """
    if not path.exists():
        raise WatchSetupError(f"cannot watch {path}: file does not exist")

    stream_fn = event_stream if event_stream is not None else _default_event_stream
    for batch in stream_fn(path, stop_event=stop_event):
        if filter_events_for_path(batch, path):
            on_change()
