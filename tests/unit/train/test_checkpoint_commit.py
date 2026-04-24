"""Two-phase checkpoint commit — allocation, flip, crash safety."""

from __future__ import annotations

from pathlib import Path

import pytest

import dlm.train.checkpoint_commit as checkpoint_commit
from dlm.store.paths import for_dlm
from dlm.train.checkpoint_commit import (
    _uniquify_rejected,
    allocate_next_version,
    commit_version,
    fsync_dir,
    list_pending_versions,
)
from dlm.train.integrity import NaNWeightsError


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

    def test_ignores_non_dir_entries(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        (store.adapter_versions / "v0009").write_text("not a directory")
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

    def test_nonfinite_writer_uniquify_failure_leaves_pending(self, tmp_path: Path, monkeypatch) -> None:
        store = _store(tmp_path)

        def bad_writer(p: Path) -> None:
            (p / "weights.safetensors").write_text("bad")
            raise NaNWeightsError(["adapter.lora_A"])

        def boom(_: Path) -> Path:
            raise RuntimeError("no rejected slot")

        monkeypatch.setattr(checkpoint_commit, "_uniquify_rejected", boom)

        with pytest.raises(RuntimeError, match="no rejected slot"):
            commit_version(store, bad_writer)

        assert store.adapter_version(1).exists()
        assert store.resolve_current_adapter() is None

    def test_nonfinite_writer_rename_failure_still_reraises(self, tmp_path: Path, monkeypatch) -> None:
        store = _store(tmp_path)

        def bad_writer(p: Path) -> None:
            (p / "weights.safetensors").write_text("bad")
            raise NaNWeightsError(["adapter.lora_B"])

        def bad_rename(self: Path, target: Path) -> Path:
            raise OSError("rename blocked")

        monkeypatch.setattr(Path, "rename", bad_rename)

        with pytest.raises(NaNWeightsError, match="NaN/inf"):
            commit_version(store, bad_writer)

        assert store.adapter_version(1).exists()
        assert store.resolve_current_adapter() is None

    def test_nonfinite_writer_renames_to_rejected_path(self, tmp_path: Path) -> None:
        store = _store(tmp_path)

        def bad_writer(p: Path) -> None:
            (p / "weights.safetensors").write_text("bad")
            raise NaNWeightsError(["adapter.lora_B"])

        with pytest.raises(NaNWeightsError, match="NaN/inf"):
            commit_version(store, bad_writer)

        assert not store.adapter_version(1).exists()
        assert (store.adapter_versions / "v0001-rejected").exists()
        assert store.resolve_current_adapter() is None


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

    def test_named_adapter_pending_versions_report_orphans(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        orphan = allocate_next_version(store, adapter_name="writer")
        commit_version(store, lambda p: (p / "a").write_text("a"), adapter_name="writer")
        pending = list_pending_versions(store, adapter_name="writer")
        assert [p.name for p in pending] == [orphan.name]

    def test_named_adapter_pending_versions_without_current(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        orphan = allocate_next_version(store, adapter_name="writer")
        pending = list_pending_versions(store, adapter_name="writer")
        assert pending == [orphan]


class TestFsyncDir:
    def test_fsync_no_error_on_real_dir(self, tmp_path: Path) -> None:
        """fsync_dir is side-effectful; just assert it doesn't raise."""
        fsync_dir(tmp_path)


class TestRejectedPathAllocation:
    def test_returns_first_available_suffix(self, tmp_path: Path) -> None:
        pending = tmp_path / "v0001"
        pending.mkdir()
        (tmp_path / "v0001-rejected").mkdir()
        (tmp_path / "v0001-rejected-1").mkdir()
        assert _uniquify_rejected(pending) == tmp_path / "v0001-rejected-2"

    def test_raises_after_1000_collisions(self, tmp_path: Path) -> None:
        pending = tmp_path / "v0001"
        pending.mkdir()
        (tmp_path / "v0001-rejected").mkdir()
        for i in range(1, 1000):
            (tmp_path / f"v0001-rejected-{i}").mkdir()

        with pytest.raises(RuntimeError, match="after 1000 attempts"):
            _uniquify_rejected(pending)
