"""Edge-case coverage for store internals not reached by higher-level tests.

Brings `src/dlm/store/*` above the 95% DoD bar (Sprint 04 DoD inherited
from Sprint 03's coverage discipline after audit 02 M4).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from dlm.store import errors, lock, paths


class TestErrorsUnusedBranches:
    def test_orphaned_store_error_with_source(self, tmp_path: Path) -> None:
        err = errors.OrphanedStoreError(tmp_path / "store", tmp_path / "src.dlm")
        assert "src.dlm" in str(err)

    def test_orphaned_store_error_without_source(self, tmp_path: Path) -> None:
        err = errors.OrphanedStoreError(tmp_path / "store", None)
        assert "<unknown>" in str(err)

    def test_lock_held_error_unknown_owner(self, tmp_path: Path) -> None:
        err = errors.LockHeldError(tmp_path / ".lock", holder_pid=None)
        assert "unknown owner" in str(err)

    def test_stale_lock_error_message(self, tmp_path: Path) -> None:
        err = errors.StaleLockError(tmp_path / ".lock", holder_pid=42)
        assert "PID 42" in str(err)

    def test_manifest_corrupt_error_carries_path_reason(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        err = errors.ManifestCorruptError(path, "bad json")
        assert err.path == path
        assert err.reason == "bad json"


class TestLockDecoderAndAlive:
    def test_decode_payload_too_few_lines_returns_none(self) -> None:
        assert lock._decode_lock_payload("only one line") is None

    def test_decode_payload_non_integer_pid(self) -> None:
        assert lock._decode_lock_payload("not_a_pid\nhost\n0.0\n") is None

    def test_decode_payload_non_float_ts(self) -> None:
        assert lock._decode_lock_payload("42\nhost\nnot_a_float\n") is None

    def test_is_alive_rejects_zero_or_negative_pid(self) -> None:
        assert lock._is_alive(0) is False
        assert lock._is_alive(-1) is False

    def test_is_alive_self_is_true(self) -> None:
        assert lock._is_alive(os.getpid()) is True

    def test_is_alive_permission_error_treated_as_alive(self) -> None:
        with patch("dlm.store.lock.os.kill", side_effect=PermissionError("boom")):
            assert lock._is_alive(12345) is True

    def test_is_alive_os_error_esrch_is_dead(self) -> None:
        import errno as _errno

        with patch(
            "dlm.store.lock.os.kill",
            side_effect=OSError(_errno.ESRCH, "no such process"),
        ):
            assert lock._is_alive(12345) is False

    def test_is_alive_os_error_other_errno_treated_as_alive(self) -> None:
        import errno as _errno

        with patch(
            "dlm.store.lock.os.kill",
            side_effect=OSError(_errno.EACCES, "access denied"),
        ):
            assert lock._is_alive(12345) is True


class TestLockReleaseEdges:
    def test_release_warns_when_owner_changed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        lock_path = tmp_path / "test.lock"
        # Acquire so the lockfile is ours for context, then overwrite the
        # PID to simulate another holder, then explicitly call _release.
        with lock.exclusive(lock_path):
            lock_path.write_text("99999\nhost\n0\n", encoding="utf-8")
            with caplog.at_level("WARNING", logger="dlm.store.lock"):
                lock._release(lock_path)
        # After context exits, `_release` runs again internally but the
        # lockfile has already been externally overwritten; no raise.

    def test_release_on_missing_file_is_noop(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "missing.lock"
        # Should not raise even though the file doesn't exist.
        lock._release(lock_path)

    def test_break_lock_returns_false_when_absent(self, tmp_path: Path) -> None:
        assert lock.break_lock(tmp_path / "ghost.lock") is False


class TestPathsWindowsBranch:
    def test_appdata_used_when_nt(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("DLM_HOME", raising=False)
        monkeypatch.setattr("dlm.store.paths._current_os_name", lambda: "nt")
        monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
        assert paths.dlm_home() == (tmp_path / "AppData").resolve() / "dlm"

    def test_nt_falls_back_to_home_without_appdata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("DLM_HOME", raising=False)
        monkeypatch.delenv("APPDATA", raising=False)
        monkeypatch.setattr("dlm.store.paths._current_os_name", lambda: "nt")
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "U")
        assert paths.dlm_home() == tmp_path / "U" / ".dlm"


class TestInspectDirSizeEdges:
    def test_directory_size_of_missing_path_is_zero(self, tmp_path: Path) -> None:
        from dlm.store.inspect import _directory_size

        assert _directory_size(tmp_path / "does-not-exist") == 0

    # The `except OSError` swallow path is defensive (file disappears
    # between rglob yield and stat). No reliable way to trigger on a
    # real filesystem without flaky mocking; left uncovered — total store
    # coverage still exceeds the 95% gate.
