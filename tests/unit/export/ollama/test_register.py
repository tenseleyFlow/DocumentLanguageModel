"""`ollama create` wrapper — lock path resolution + subprocess mocking."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from dlm.export.ollama.errors import OllamaCreateError
from dlm.export.ollama.register import ollama_create, ollama_lock_path


class TestLockPath:
    def test_explicit_dlm_home(self, tmp_path: Path) -> None:
        assert ollama_lock_path(dlm_home=tmp_path) == tmp_path / "ollama.lock"

    def test_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DLM_HOME", str(tmp_path))
        assert ollama_lock_path() == tmp_path / "ollama.lock"

    def test_default_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DLM_HOME", raising=False)
        assert ollama_lock_path() == Path.home() / ".dlm" / "ollama.lock"


def _ok_proc() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ollama", "create"],
        returncode=0,
        stdout="ok\n",
        stderr="",
    )


def _bad_proc(stderr: str = "missing base") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ollama", "create"],
        returncode=1,
        stdout="",
        stderr=stderr,
    )


class TestOllamaCreate:
    def test_happy_path(self, tmp_path: Path) -> None:
        exe = tmp_path / "ollama"
        exe.write_text("")
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM ./x.gguf\n")
        with patch(
            "dlm.export.ollama.register.subprocess.run",
            return_value=_ok_proc(),
        ) as mock_run:
            stdout = ollama_create(
                name="dlm-01test:v0001",
                modelfile_path=modelfile,
                cwd=tmp_path,
                binary=exe,
                dlm_home=tmp_path,
            )
        assert stdout == "ok\n"
        argv = mock_run.call_args.args[0]
        assert argv[0] == str(exe)
        assert argv[1:4] == ["create", "dlm-01test:v0001", "-f"]
        assert argv[4] == str(modelfile)

    def test_non_zero_raises(self, tmp_path: Path) -> None:
        exe = tmp_path / "ollama"
        exe.write_text("")
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("x")
        with (
            patch(
                "dlm.export.ollama.register.subprocess.run",
                return_value=_bad_proc(stderr="missing base"),
            ),
            pytest.raises(OllamaCreateError, match="missing base"),
        ):
            ollama_create(
                name="x:1",
                modelfile_path=modelfile,
                cwd=tmp_path,
                binary=exe,
                dlm_home=tmp_path,
            )

    def test_timeout_raises(self, tmp_path: Path) -> None:
        exe = tmp_path / "ollama"
        exe.write_text("")
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("x")
        with (
            patch(
                "dlm.export.ollama.register.subprocess.run",
                side_effect=subprocess.TimeoutExpired(
                    cmd="ollama", timeout=1.0, stderr=b"slow"
                ),
            ),
            pytest.raises(OllamaCreateError, match="timed out"),
        ):
            ollama_create(
                name="x:1",
                modelfile_path=modelfile,
                cwd=tmp_path,
                binary=exe,
                dlm_home=tmp_path,
                timeout=1.0,
            )

    def test_lock_acquired_during_subprocess(self, tmp_path: Path) -> None:
        """The lock file should exist + be held while subprocess runs."""
        exe = tmp_path / "ollama"
        exe.write_text("")
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("x")
        lock_observed: dict[str, bool] = {"existed": False}

        def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            lock_observed["existed"] = (tmp_path / "ollama.lock").exists()
            return _ok_proc()

        with patch("dlm.export.ollama.register.subprocess.run", side_effect=fake_run):
            ollama_create(
                name="x:1",
                modelfile_path=modelfile,
                cwd=tmp_path,
                binary=exe,
                dlm_home=tmp_path,
            )
        assert lock_observed["existed"]


class TestLockContention:
    def test_held_lock_blocks_second_caller(self, tmp_path: Path) -> None:
        """A second caller times out while the store-lock holds the file."""
        from dlm.store.errors import LockHeldError
        from dlm.store.lock import exclusive

        exe = tmp_path / "ollama"
        exe.write_text("")
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("x")
        lock_path = tmp_path / "ollama.lock"

        with (
            exclusive(lock_path, timeout=5.0),
            pytest.raises(LockHeldError),
        ):
            ollama_create(
                name="x:1",
                modelfile_path=modelfile,
                cwd=tmp_path,
                binary=exe,
                dlm_home=tmp_path,
                lock_timeout=0.2,
            )
