"""write_lock / load_lock — atomic I/O and version gating."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dlm.lock.errors import LockSchemaError
from dlm.lock.schema import LOCK_FILENAME, DlmLock
from dlm.lock.writer import load_lock, lock_path, write_lock


def _lock(**overrides: object) -> DlmLock:
    base = {
        "created_at": datetime(2026, 4, 19, 12, 0, 0, tzinfo=UTC),
        "dlm_id": "01HZXY",
        "dlm_sha256": "a" * 64,
        "base_model_revision": "rev1",
        "hardware_tier": "cpu",
        "seed": 42,
        "determinism_class": "best-effort",
        "last_run_id": 1,
    }
    base.update(overrides)
    return DlmLock(**base)  # type: ignore[arg-type]


class TestLockPath:
    def test_returns_dlm_lock_filename_under_root(self, tmp_path: Path) -> None:
        assert lock_path(tmp_path) == tmp_path / LOCK_FILENAME


class TestWriteLock:
    def test_writes_readable_json(self, tmp_path: Path) -> None:
        written = write_lock(tmp_path, _lock())
        assert written.is_file()
        text = written.read_text(encoding="utf-8")
        assert text.endswith("\n")
        # Stable-sorted keys for byte-identical re-writes.
        assert text.index('"created_at"') < text.index('"seed"')

    def test_round_trip_equal(self, tmp_path: Path) -> None:
        original = _lock(pinned_versions={"torch": "2.5.1"})
        write_lock(tmp_path, original)
        loaded = load_lock(tmp_path)
        assert loaded == original

    def test_overwrite_replaces_prior_lock(self, tmp_path: Path) -> None:
        write_lock(tmp_path, _lock(seed=1))
        write_lock(tmp_path, _lock(seed=99))
        loaded = load_lock(tmp_path)
        assert loaded is not None
        assert loaded.seed == 99


class TestLoadLock:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_lock(tmp_path) is None

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / LOCK_FILENAME).write_text("{not valid")
        with pytest.raises(LockSchemaError, match="invalid JSON"):
            load_lock(tmp_path)

    def test_non_object_top_level_raises(self, tmp_path: Path) -> None:
        (tmp_path / LOCK_FILENAME).write_text("[]")
        with pytest.raises(LockSchemaError, match="must be an object"):
            load_lock(tmp_path)

    def test_newer_version_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / LOCK_FILENAME).write_text('{"lock_version": 99}')
        with pytest.raises(LockSchemaError, match="unsupported lock_version"):
            load_lock(tmp_path)

    def test_schema_violation_surfaces_as_lock_schema_error(self, tmp_path: Path) -> None:
        (tmp_path / LOCK_FILENAME).write_text('{"lock_version": 1, "dlm_id": ""}')
        with pytest.raises(LockSchemaError, match="schema validation"):
            load_lock(tmp_path)
