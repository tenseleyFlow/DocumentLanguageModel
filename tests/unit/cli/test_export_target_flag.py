"""`dlm export --target` parsing + validation (Sprint 41 substrate)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def _joined(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(text.split())


def _scaffold_doc(tmp_path: Path) -> Path:
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


class TestExportTargetFlag:
    def test_unknown_target_rejected_before_export_work(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "export",
                str(tmp_path / "ghost.dlm"),
                "--target",
                "vllm",
            ],
        )
        assert result.exit_code == 2
        text = _joined(result)
        assert "unknown export target" in text
        assert "ollama" in text
        assert "llama-server" in text

    def test_ollama_target_reaches_existing_mutex_validation(self, tmp_path: Path) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "export",
                str(doc),
                "--target",
                "ollama",
                "--draft",
                "qwen2.5:0.5b",
                "--no-draft",
            ],
        )
        assert result.exit_code == 2
        assert "mutually exclusive" in _joined(result)

    def test_llama_server_requires_no_smoke_for_now(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "export",
                str(tmp_path / "ghost.dlm"),
                "--target",
                "llama-server",
            ],
        )
        assert result.exit_code == 2
        assert "--no-smoke" in _joined(result)
