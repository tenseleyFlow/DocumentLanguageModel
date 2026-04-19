"""`dlm train --strict-lock` / `--update-lock` / `--ignore-lock` (Sprint 15)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def _write_minimal_dlm(path: Path) -> None:
    path.write_text(
        "---\n"
        "dlm_id: 01TEST0" + "0" * 19 + "\n"
        "base_model: smollm2-135m\n"
        "training:\n"
        "  seed: 42\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )


class TestLockFlagMutex:
    def test_strict_and_update_together_exits_2(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "train",
                str(doc),
                "--strict-lock",
                "--update-lock",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "mutually exclusive" in result.output

    def test_strict_and_ignore_together_exits_2(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "train",
                str(doc),
                "--strict-lock",
                "--ignore-lock",
            ],
        )
        assert result.exit_code == 2

    def test_update_and_ignore_together_exits_2(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "train",
                str(doc),
                "--update-lock",
                "--ignore-lock",
            ],
        )
        assert result.exit_code == 2

    def test_all_three_flags_together_exits_2(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "train",
                str(doc),
                "--strict-lock",
                "--update-lock",
                "--ignore-lock",
            ],
        )
        assert result.exit_code == 2
