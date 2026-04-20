"""Event-filter logic in watcher.filter_events_for_path."""

from __future__ import annotations

from pathlib import Path

from dlm.watch.watcher import filter_events_for_path


class TestFilterEvents:
    def test_matches_target(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        target.write_text("x")
        batch: set[tuple[object, str]] = {("modified", str(target))}
        assert filter_events_for_path(batch, target) is True

    def test_ignores_other_files(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        other = tmp_path / "other.dlm"
        target.write_text("x")
        other.write_text("y")
        batch: set[tuple[object, str]] = {("modified", str(other))}
        assert filter_events_for_path(batch, target) is False

    def test_matches_among_unrelated_events(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        other = tmp_path / "other.dlm"
        target.write_text("x")
        other.write_text("y")
        batch: set[tuple[object, str]] = {
            ("modified", str(other)),
            ("modified", str(target)),
        }
        assert filter_events_for_path(batch, target) is True

    def test_empty_batch_is_no_match(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        target.write_text("x")
        batch: set[tuple[object, str]] = set()
        assert filter_events_for_path(batch, target) is False

    def test_nonexistent_path_still_matched_by_string(self, tmp_path: Path) -> None:
        """Atomic rename temporarily makes the path missing. The filter
        must still match by string rather than silently return False."""
        target = tmp_path / "gone.dlm"
        batch: set[tuple[object, str]] = {("added", str(target))}
        assert filter_events_for_path(batch, target) is True
