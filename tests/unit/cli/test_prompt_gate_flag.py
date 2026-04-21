"""CLI flag validation for `dlm prompt --gate`."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app

_FRONTMATTER = (
    "---\ndlm_id: 01KPQ8GATECM10000000000000\ndlm_version: 8\nbase_model: smollm2-135m\n---\n"
)


def _write_dlm(path: Path) -> None:
    path.write_text(_FRONTMATTER + "prose\n", encoding="utf-8")


class TestGateFlag:
    def test_invalid_gate_value_refused(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        _write_dlm(doc)
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(tmp_path), "prompt", str(doc), "hello", "--gate", "bogus"],
        )
        assert result.exit_code == 2, result.output
        assert "--gate must be" in result.output
