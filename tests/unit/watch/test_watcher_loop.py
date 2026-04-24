"""Loop-level coverage for watch_for_changes and default stream wrapper."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dlm.watch.errors import WatchSetupError
from dlm.watch.watcher import _default_event_stream, watch_for_changes


def test_default_event_stream_wraps_watchfiles_iterator(tmp_path: Path) -> None:
    seen: list[tuple[str, object | None]] = []
    expected = [{("modified", str(tmp_path / "doc.dlm"))}]

    def fake_watch(
        path: str, *, stop_event: object | None = None
    ) -> Iterator[set[tuple[object, str]]]:
        seen.append((path, stop_event))
        yield from expected

    with patch.dict("sys.modules", {"watchfiles": SimpleNamespace(watch=fake_watch)}):
        batches = list(_default_event_stream(tmp_path / "doc.dlm", stop_event="stop"))

    assert batches == expected
    assert seen == [(str(tmp_path), "stop")]


def test_watch_for_changes_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(WatchSetupError, match="does not exist"):
        watch_for_changes(tmp_path / "missing.dlm", lambda: None)


def test_watch_for_changes_invokes_callback_for_matching_batches(tmp_path: Path) -> None:
    target = tmp_path / "doc.dlm"
    target.write_text("x")
    seen: list[str] = []

    def event_stream(
        _path: Path, *, stop_event: object | None = None
    ) -> Iterator[set[tuple[object, str]]]:
        assert stop_event == "stop"
        yield {("modified", str(tmp_path / "other.dlm"))}
        yield {("modified", str(target))}
        yield {("added", str(target))}

    watch_for_changes(
        target, lambda: seen.append("changed"), stop_event="stop", event_stream=event_stream
    )

    assert seen == ["changed", "changed"]
