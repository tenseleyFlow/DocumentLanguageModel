"""Audit-03 cross-cutting: `dlm.io.atomic` atomic-write helper.

Covers the replacement of three near-identical copies (io/text.py,
store/manifest.py, store/paths.py) with a single primitive. Every prior
caller's behavior already has a test in its own module; these tests pin
down the `atomic` module's direct contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from dlm.io import atomic


class TestWriteBytes:
    def test_replaces_target_atomically(self, tmp_path: Path) -> None:
        target = tmp_path / "file.bin"
        atomic.write_bytes(target, b"hello")
        assert target.read_bytes() == b"hello"
        # No tmp siblings left behind.
        assert list(tmp_path.iterdir()) == [target]

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "file.bin"
        target.write_bytes(b"old")
        atomic.write_bytes(target, b"new")
        assert target.read_bytes() == b"new"
        assert list(tmp_path.iterdir()) == [target]


class TestWriteText:
    def test_utf8_default(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.txt"
        atomic.write_text(target, "hello — world")
        assert target.read_bytes() == "hello — world".encode()

    def test_custom_encoding(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.txt"
        atomic.write_text(target, "hello", encoding="ascii")
        assert target.read_bytes() == b"hello"


class TestCleanupStaleTmp:
    def test_removes_only_dead_pid_tmp_files(self, tmp_path: Path) -> None:
        live = tmp_path / "real.txt.tmp.1"
        dead = tmp_path / "real.txt.tmp.99999999"
        live.write_bytes(b"live")
        dead.write_bytes(b"dead")

        def fake_is_alive(pid: int) -> bool:
            return pid == 1

        with patch("dlm.io.atomic._is_alive", side_effect=fake_is_alive):
            removed = atomic.cleanup_stale_tmp_files(tmp_path)

        assert removed == [dead]
        assert live.exists()
        assert not dead.exists()

    def test_skips_non_tmp_files(self, tmp_path: Path) -> None:
        real = tmp_path / "manifest.json"
        real.write_bytes(b"{}")
        assert atomic.cleanup_stale_tmp_files(tmp_path) == []
        assert real.exists()

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        assert atomic.cleanup_stale_tmp_files(tmp_path / "absent") == []

    def test_malformed_tmp_suffix_ignored(self, tmp_path: Path) -> None:
        malformed = tmp_path / "name.tmp.notapid"
        malformed.write_bytes(b"x")
        assert atomic.cleanup_stale_tmp_files(tmp_path) == []
        assert malformed.exists()


class TestIsAlive:
    def test_zero_or_negative_dead(self) -> None:
        assert atomic._is_alive(0) is False
        assert atomic._is_alive(-1) is False

    def test_self_is_alive(self) -> None:
        assert atomic._is_alive(os.getpid()) is True

    def test_dead_pid_is_dead(self) -> None:
        # 99999999 is almost certainly not live.
        assert atomic._is_alive(99999999) is False

    def test_permission_error_treated_as_alive(self) -> None:
        with patch("dlm.io.atomic.os.kill", side_effect=PermissionError):
            assert atomic._is_alive(12345) is True

    def test_generic_os_error_treated_as_dead(self) -> None:
        with patch("dlm.io.atomic.os.kill", side_effect=OSError):
            assert atomic._is_alive(12345) is False
