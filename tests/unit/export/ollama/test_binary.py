"""`locate_ollama` + version parsing + min-version gate (audit F16)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from dlm.export.ollama.binary import (
    OLLAMA_MIN_VERSION,
    check_ollama_version,
    locate_ollama,
    ollama_version,
)
from dlm.export.ollama.errors import (
    OllamaBinaryNotFoundError,
    OllamaVersionError,
)


class TestLocate:
    def test_override_used_when_file_exists(self, tmp_path: Path) -> None:
        fake = tmp_path / "ollama"
        fake.write_text("#!/bin/sh\necho 0.4.2\n")
        fake.chmod(0o755)
        assert locate_ollama(override=fake) == fake

    def test_override_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OllamaBinaryNotFoundError):
            locate_ollama(override=tmp_path / "missing")

    def test_path_lookup(self, tmp_path: Path) -> None:
        """`shutil.which` hit takes precedence over standard paths."""
        fake = tmp_path / "ollama"
        fake.write_text("")
        with patch("dlm.export.ollama.binary.shutil.which", return_value=str(fake)):
            assert locate_ollama() == fake

    def test_standard_paths_fallback(self, tmp_path: Path) -> None:
        fake = tmp_path / "ollama"
        fake.write_text("")
        with (
            patch("dlm.export.ollama.binary.shutil.which", return_value=None),
            patch("dlm.export.ollama.binary._STANDARD_PATHS", (fake,)),
        ):
            assert locate_ollama() == fake

    def test_not_found_raises_with_install_link(self) -> None:
        with (
            patch("dlm.export.ollama.binary.shutil.which", return_value=None),
            patch("dlm.export.ollama.binary._STANDARD_PATHS", ()),
            pytest.raises(OllamaBinaryNotFoundError, match="ollama.com/download"),
        ):
            locate_ollama()


class TestVersionParse:
    def _fake_proc(self, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["ollama", "--version"],
            returncode=0,
            stdout=stdout,
            stderr=stderr,
        )

    def test_parses_is_format(self, tmp_path: Path) -> None:
        fake = tmp_path / "ollama"
        fake.write_text("")
        with patch(
            "dlm.export.ollama.binary.subprocess.run",
            return_value=self._fake_proc(stdout="ollama version is 0.4.2\n"),
        ):
            assert ollama_version(binary=fake) == (0, 4, 2)

    def test_parses_plain(self, tmp_path: Path) -> None:
        fake = tmp_path / "ollama"
        fake.write_text("")
        with patch(
            "dlm.export.ollama.binary.subprocess.run",
            return_value=self._fake_proc(stdout="0.5.11\n"),
        ):
            assert ollama_version(binary=fake) == (0, 5, 11)

    def test_parses_from_stderr(self, tmp_path: Path) -> None:
        fake = tmp_path / "ollama"
        fake.write_text("")
        with patch(
            "dlm.export.ollama.binary.subprocess.run",
            return_value=self._fake_proc(stdout="", stderr="ollama version 0.6.0"),
        ):
            assert ollama_version(binary=fake) == (0, 6, 0)

    def test_unparseable_raises(self, tmp_path: Path) -> None:
        fake = tmp_path / "ollama"
        fake.write_text("")
        with (
            patch(
                "dlm.export.ollama.binary.subprocess.run",
                return_value=self._fake_proc(stdout="not a version"),
            ),
            pytest.raises(OllamaVersionError),
        ):
            ollama_version(binary=fake)

    def test_binary_missing_raises(self, tmp_path: Path) -> None:
        fake = tmp_path / "ollama"
        fake.write_text("")
        with (
            patch(
                "dlm.export.ollama.binary.subprocess.run",
                side_effect=FileNotFoundError("gone"),
            ),
            pytest.raises(OllamaBinaryNotFoundError),
        ):
            ollama_version(binary=fake)


class TestMinVersionGate:
    def test_below_minimum_raises(self, tmp_path: Path) -> None:
        fake = tmp_path / "ollama"
        fake.write_text("")
        with (
            patch(
                "dlm.export.ollama.binary.ollama_version",
                return_value=(0, 3, 99),
            ),
            pytest.raises(OllamaVersionError) as excinfo,
        ):
            check_ollama_version(binary=fake)
        assert excinfo.value.detected == (0, 3, 99)
        assert excinfo.value.required == OLLAMA_MIN_VERSION

    def test_exactly_minimum_passes(self, tmp_path: Path) -> None:
        fake = tmp_path / "ollama"
        fake.write_text("")
        with patch(
            "dlm.export.ollama.binary.ollama_version",
            return_value=OLLAMA_MIN_VERSION,
        ):
            assert check_ollama_version(binary=fake) == OLLAMA_MIN_VERSION

    def test_above_minimum_passes(self, tmp_path: Path) -> None:
        fake = tmp_path / "ollama"
        fake.write_text("")
        bumped = (OLLAMA_MIN_VERSION[0], OLLAMA_MIN_VERSION[1] + 1, 0)
        with patch(
            "dlm.export.ollama.binary.ollama_version",
            return_value=bumped,
        ):
            assert check_ollama_version(binary=fake) == bumped
