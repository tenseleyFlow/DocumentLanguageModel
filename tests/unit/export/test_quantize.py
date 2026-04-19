"""`run_checked` — timeout, stderr tail, SubprocessError wrapping."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from dlm.export.errors import SubprocessError
from dlm.export.quantize import run_checked


class TestRunChecked:
    def test_happy_path_returns_completed(self) -> None:
        """`true` always succeeds with empty output."""
        proc = run_checked(["true"])
        assert proc.returncode == 0

    def test_non_zero_exit_raises_subprocess_error(self) -> None:
        with pytest.raises(SubprocessError) as excinfo:
            run_checked(["false"])
        assert excinfo.value.returncode == 1

    def test_timeout_raises_subprocess_error_with_none_returncode(self) -> None:
        """Simulated timeout: sleep 10, kill at 0.1s."""
        with pytest.raises(SubprocessError) as excinfo:
            run_checked(["sleep", "10"], timeout=0.1)
        assert excinfo.value.returncode is None
        # Message mentions timeout/None-ish
        assert "timed out" in str(excinfo.value)

    def test_stderr_tail_included_in_error(self) -> None:
        """Shell out to a command with predictable stderr."""
        with pytest.raises(SubprocessError) as excinfo:
            # bash prints an error message when invoking a nonexistent flag.
            run_checked(["sh", "-c", "echo oops >&2; exit 7"])
        assert excinfo.value.returncode == 7
        assert "oops" in excinfo.value.stderr_tail

    def test_stderr_tail_truncated_to_4kb(self) -> None:
        """Massive stderr is tailed to the last ~4 KB."""
        # Generate 10 KB of stderr then exit non-zero.
        script = "python3 -c 'import sys; sys.stderr.write(\"x\"*10000); sys.exit(2)'"
        with pytest.raises(SubprocessError) as excinfo:
            run_checked(["sh", "-c", script])
        # Tail is ≤4096 chars (the bytes; UTF-8 for 'x' is 1 byte).
        assert len(excinfo.value.stderr_tail) <= 4096


class TestCmdPropagation:
    def test_cmd_captured_in_error(self) -> None:
        with pytest.raises(SubprocessError) as excinfo:
            run_checked(["sh", "-c", "exit 3"])
        assert excinfo.value.cmd[0] == "sh"
        assert excinfo.value.returncode == 3


class TestTypedReraise:
    def test_keyboard_interrupt_not_wrapped(self) -> None:
        """Timeouts become SubprocessError; KeyboardInterrupt does not."""
        with (
            patch("dlm.export.quantize.subprocess.run", side_effect=KeyboardInterrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            run_checked(["true"])

    def test_generic_subprocess_timeout_mapped(self) -> None:
        with (
            patch(
                "dlm.export.quantize.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1.0, stderr=b"late"),
            ),
            pytest.raises(SubprocessError) as excinfo,
        ):
            run_checked(["x"])
        assert excinfo.value.returncode is None
        assert "late" in excinfo.value.stderr_tail
