"""Two-phase checkpoint commit — allocation, flip, crash safety."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.store.paths import for_dlm
from dlm.train.checkpoint_commit import (
    allocate_next_version,
    commit_version,
    fsync_dir,
    list_pending_versions,
)


def _store(home: Path):
    store = for_dlm("01TEST", home=home)
    store.ensure_layout()
    return store


class TestAllocation:
    def test_first_version_is_v0001(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        v1 = allocate_next_version(store)
        assert v1.name == "v0001"
        assert v1.exists()

    def test_allocates_sequentially(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        v1 = allocate_next_version(store)
        v2 = allocate_next_version(store)
        v3 = allocate_next_version(store)
        assert v2.name == "v0002"
        assert v3.name == "v0003"
        # Non-reuse: allocating over an existing dir is rejected.
        assert v1.name != v2.name

    def test_ignores_non_version_dirs(self, tmp_path: Path) -> None:
        """Stray dirs (e.g., tmp/ or backup/) don't advance the counter."""
        store = _store(tmp_path)
        (store.adapter_versions / "scratch").mkdir()
        (store.adapter_versions / "v-not-a-number").mkdir()
        v1 = allocate_next_version(store)
        assert v1.name == "v0001"


class TestCommitVersion:
    def test_happy_path_flips_current(self, tmp_path: Path) -> None:
        store = _store(tmp_path)

        def writer(p: Path) -> None:
            (p / "adapter_config.json").write_text("{}")

        committed = commit_version(store, writer)
        assert committed.name == "v0001"
        assert store.resolve_current_adapter() == committed

    def test_writer_exception_preserves_current(self, tmp_path: Path) -> None:
        """Simulated crash-during-write: old current stays authoritative."""
        store = _store(tmp_path)

        # First, do a successful commit so `current` has something to preserve.
        v1 = commit_version(store, lambda p: (p / "ok").write_text("ok"))
        assert store.resolve_current_adapter() == v1

        # Now a writer that crashes.
        def bad_writer(p: Path) -> None:
            (p / "partial").write_text("half")
            raise RuntimeError("SIGKILL-like")

        with pytest.raises(RuntimeError, match="SIGKILL-like"):
            commit_version(store, bad_writer)

        # Current pointer is unchanged.
        assert store.resolve_current_adapter() == v1

    def test_pending_dir_remains_after_crash(self, tmp_path: Path) -> None:
        """Left on disk for post-mortem; next allocate skips over it."""
        store = _store(tmp_path)

        def bad_writer(p: Path) -> None:
            (p / "x").write_text("x")
            raise RuntimeError("crash")

        with pytest.raises(RuntimeError):
            commit_version(store, bad_writer)

        # v0001 should exist (even though it's not current).
        v1_path = store.adapter_version(1)
        assert v1_path.exists()

        # Next allocate goes to v0002.
        v2 = allocate_next_version(store)
        assert v2.name == "v0002"


class TestListPending:
    def test_no_pending_when_all_committed(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        v1 = commit_version(store, lambda p: (p / "a").write_text("a"))
        pending = list_pending_versions(store)
        assert pending == []
        assert store.resolve_current_adapter() == v1

    def test_reports_pre_current_versions(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        # Orphan v0001
        v1 = allocate_next_version(store)
        # Committed v0002
        commit_version(store, lambda p: (p / "a").write_text("a"))
        pending = list_pending_versions(store)
        assert [p.name for p in pending] == [v1.name]


class TestFsyncDir:
    def test_fsync_no_error_on_real_dir(self, tmp_path: Path) -> None:
        """fsync_dir is side-effectful; just assert it doesn't raise."""
        fsync_dir(tmp_path)
