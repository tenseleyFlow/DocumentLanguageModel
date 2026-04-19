"""`dlm export` CLI flag mutex checks (Sprint 12.5)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def _joined_output(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(text.split())


def _scaffold_doc(tmp_path: Path) -> Path:
    """Minimal .dlm file for the CLI to parse."""
    doc = tmp_path / "mydoc.dlm"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_path / "home"),
            "init",
            str(doc),
            "--base",
            "smollm2-135m",
        ],
    )
    assert result.exit_code == 0, result.output
    return doc


class TestDraftFlagMutex:
    """Sprint 12.5: --draft and --no-draft are mutually exclusive."""

    def test_both_flags_exits_2(self, tmp_path: Path) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "export",
                str(doc),
                "--draft",
                "qwen2.5:0.5b",
                "--no-draft",
            ],
        )
        assert result.exit_code == 2
        assert "mutually exclusive" in _joined_output(result)
