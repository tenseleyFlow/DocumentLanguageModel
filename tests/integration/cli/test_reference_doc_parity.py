"""Audit-11 M2 — keep the hand-maintained CLI reference aligned with the live CLI."""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app

_RUNNER = CliRunner()
_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_DOC = (_ROOT / "docs" / "cli" / "reference.md").read_text(encoding="utf-8")


def _normalized_help(*argv: str) -> str:
    result = _RUNNER.invoke(app, [*argv, "--help"])
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def _section(name: str) -> str:
    pattern = rf"### `dlm {re.escape(name)}`\n\n(.*?)(?=\n### `dlm |\Z)"
    match = re.search(pattern, _REFERENCE_DOC, re.S)
    assert match is not None, f"missing `dlm {name}` section in docs/cli/reference.md"
    return match.group(1)


def test_reference_doc_covers_audited_train_flags() -> None:
    section = _section("train")
    help_text = _normalized_help("train")
    for flag in (
        "--base",
        "--include",
        "--exclude",
        "--recursive",
        "--no-recursive",
        "--name",
        "--policy",
        "--rescaffold",
        "--phase",
        "--strict-metrics",
        "--skip-export-probes",
    ):
        assert flag in help_text
        assert flag in section


def test_reference_doc_covers_audio_and_verify_surface() -> None:
    init_section = _section("init")
    prompt_section = _section("prompt")
    verify_section = _section("verify")

    assert "--audio" in _normalized_help("init")
    assert "--audio" in init_section

    assert "--audio" in _normalized_help("prompt")
    assert "--audio" in prompt_section

    assert "--trust-on-first-use" in _normalized_help("verify")
    assert "--trust-on-first-use" in verify_section


def test_reference_doc_uses_actual_metrics_watch_order() -> None:
    section = _section("metrics")
    assert "dlm metrics <path> watch [--poll-seconds N]" in section
    assert "dlm metrics watch <path>" not in section
