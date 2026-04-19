"""`ollama run` smoke wrapper + `first_line` helper."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from dlm.export.ollama.errors import OllamaSmokeError
from dlm.export.ollama.smoke import first_line, ollama_run


def _proc(
    *, returncode: int = 0, stdout: str = "hello, world\n", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ollama", "run"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class TestOllamaRun:
    def test_happy_path(self, tmp_path: Path) -> None:
        exe = tmp_path / "ollama"
        exe.write_text("")
        with patch(
            "dlm.export.ollama.smoke.subprocess.run",
            return_value=_proc(stdout="hi there\n"),
        ):
            out = ollama_run(name="dlm-x:1", binary=exe)
        assert out == "hi there\n"

    def test_non_zero_raises(self, tmp_path: Path) -> None:
        exe = tmp_path / "ollama"
        exe.write_text("")
        with (
            patch(
                "dlm.export.ollama.smoke.subprocess.run",
                return_value=_proc(returncode=2, stderr="oops"),
            ),
            pytest.raises(OllamaSmokeError, match="oops"),
        ):
            ollama_run(name="dlm-x:1", binary=exe)

    def test_empty_stdout_raises(self, tmp_path: Path) -> None:
        """Model 'succeeded' but emitted nothing — runaway-stop guard."""
        exe = tmp_path / "ollama"
        exe.write_text("")
        with (
            patch(
                "dlm.export.ollama.smoke.subprocess.run",
                return_value=_proc(stdout="   \n"),
            ),
            pytest.raises(OllamaSmokeError, match="empty stdout"),
        ):
            ollama_run(name="dlm-x:1", binary=exe)

    def test_timeout_raises(self, tmp_path: Path) -> None:
        exe = tmp_path / "ollama"
        exe.write_text("")
        with (
            patch(
                "dlm.export.ollama.smoke.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="ollama", timeout=1.0, stderr=b"hang"),
            ),
            pytest.raises(OllamaSmokeError, match="timed out"),
        ):
            ollama_run(name="dlm-x:1", binary=exe, timeout=1.0)

    def test_argv_shape(self, tmp_path: Path) -> None:
        exe = tmp_path / "ollama"
        exe.write_text("")
        with patch(
            "dlm.export.ollama.smoke.subprocess.run",
            return_value=_proc(),
        ) as mock_run:
            ollama_run(name="dlm-x:1", prompt="ping?", binary=exe)
        argv = mock_run.call_args.args[0]
        assert argv == [str(exe), "run", "dlm-x:1", "ping?"]


class TestFirstLine:
    def test_first_non_empty_line(self) -> None:
        assert first_line("\n\n  Hello\nworld") == "Hello"

    def test_truncates_to_200_chars(self) -> None:
        line = "x" * 300
        assert len(first_line(line)) == 200

    def test_all_blank_returns_empty(self) -> None:
        assert first_line("\n\n   \n") == ""
