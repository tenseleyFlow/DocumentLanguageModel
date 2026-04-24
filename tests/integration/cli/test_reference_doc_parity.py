"""Audit-11 M2 — keep the hand-maintained CLI reference aligned with the live CLI."""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app

_RUNNER = CliRunner()
_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_DOC = (_ROOT / "docs" / "cli" / "reference.md").read_text(encoding="utf-8")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _normalized_help(*argv: str) -> str:
    result = _RUNNER.invoke(app, [*argv, "--help"])
    assert result.exit_code == 0, result.output
    plain = _ANSI_RE.sub("", result.output)
    tableless = plain.translate(str.maketrans(dict.fromkeys("│╭╮╰╯─", " ")))
    return " ".join(tableless.split())


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


def test_reference_doc_covers_export_target_surface() -> None:
    section = _section("export")
    help_text = _normalized_help("export")
    assert "--target" in help_text
    assert "--target" in section


def test_reference_doc_covers_preference_surface() -> None:
    section = _section("preference")
    help_text = _normalized_help("preference", "mine")

    for flag in ("--judge", "--threshold", "--backend", "--adapter", "--apply"):
        assert flag in help_text
        assert flag in section

    assert "dlm preference apply <path>" in section
    assert "dlm preference revert <path>" in section
    assert "dlm preference list <path>" in section


def test_reference_doc_covers_synth_surface() -> None:
    section = _section("synth")
    help_text = _normalized_help("synth", "instructions")

    for flag in (
        "--teacher",
        "--per-section",
        "--strategy",
        "--filter",
        "--threshold",
        "--apply",
        "--dry-run",
    ):
        assert flag in help_text
        assert flag in section

    assert "dlm synth preferences <path>" in section
    assert "dlm synth revert <path>" in section
    assert "dlm synth list <path>" in section
