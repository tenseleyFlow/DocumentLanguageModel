"""Smoke tests: the package imports and the CLI boots."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

import dlm
from dlm.cli.app import app


def test_package_version_is_set() -> None:
    assert dlm.__version__
    assert dlm.__version__ != "0.0.0+unknown", (
        "package must be installed via uv sync / pip install for version lookup"
    )


def test_cli_version_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"dlm {dlm.__version__}"


def test_cli_help_lists_all_v1_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    expected = {
        "init",
        "train",
        "prompt",
        "export",
        "pack",
        "unpack",
        "doctor",
        "show",
        "migrate",
    }
    for name in expected:
        assert name in result.output, f"`dlm --help` missing subcommand {name!r}"


def test_cli_subcommand_stub_raises_notimplementederror(tmp_path: Path) -> None:
    """Still-stubbed subcommands (Sprint 14's `dlm pack`) must raise with a
    sprint pointer so `dlm --help` stays self-documenting about unreleased work.

    Updated each time a stub lands: this test migrates to the *next*
    un-landed command so the smoke invariant outlives any one sprint.
    """
    runner = CliRunner()
    doc = tmp_path / "mydoc.dlm"
    result = runner.invoke(app, ["pack", str(doc)], catch_exceptions=True)
    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)
    assert "Sprint 14" in str(result.exception)


def test_python_module_entrypoint_runs() -> None:
    """`python -m dlm --version` works (packaging sanity)."""
    result = subprocess.run(
        [sys.executable, "-m", "dlm", "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == f"dlm {dlm.__version__}"
