"""Audit-11 M8 — missing help strings on prompt/export/pack/unpack flags."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app

_RUNNER = CliRunner()


def _normalized_help(*argv: str) -> str:
    result = _RUNNER.invoke(app, [*argv, "--help"])
    assert result.exit_code == 0, result.output
    tableless = result.output.translate(str.maketrans(dict.fromkeys("│╭╮╰╯─", " ")))
    return " ".join(tableless.split())


@pytest.mark.parametrize(
    ("argv", "snippets"),
    [
        (
            ("prompt",),
            (
                "--max-tokens INTEGER Max new tokens to generate.",
                "--temp FLOAT Sampling temperature.",
                "--top-p FLOAT Top-p sampling cutoff.",
            ),
        ),
        (
            ("export",),
            (
                "--merged Merge the adapter into the base before export.",
                "--dequantize Dequantize a QLoRA base to fp16 before merging.",
                "--no-smoke Register the export but skip the smoke prompt.",
            ),
        ),
        (
            ("pack",),
            (
                "--out PATH Output .dlm.pack path.",
                "--include-exports Bundle all GGUF exports into the pack.",
                "--include-base Bundle the base model snapshot",
                "--include-logs Bundle per-run JSONL logs.",
            ),
        ),
        (
            ("unpack",),
            ("--force Overwrite an existing store with the same dlm_id.",),
        ),
    ],
)
def test_help_surfaces_audited_flag_descriptions(
    argv: tuple[str, ...],
    snippets: tuple[str, ...],
) -> None:
    text = _normalized_help(*argv)
    for snippet in snippets:
        assert snippet in text
