"""`dlm prompt --adapter` flag validation (Sprint 20b)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def _joined_output(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(text.split())


def _scaffold_flat_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "flat.dlm"
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


def _scaffold_multi_doc(tmp_path: Path) -> Path:
    """Init a flat doc, then rewrite the frontmatter to add an adapters block."""
    doc = _scaffold_flat_doc(tmp_path)
    original = doc.read_text(encoding="utf-8")
    fm_end = original.find("\n---\n", original.find("---") + 3)
    assert fm_end > 0, original
    new_fm = (
        original[:fm_end]
        + "\ntraining:\n  adapters:\n    knowledge: {}\n    tone: {lora_r: 4}\n"
        + original[fm_end:]
    )
    doc.write_text(new_fm, encoding="utf-8")
    return doc


class TestFlatDocRejectsAdapter:
    def test_single_adapter_doc_with_adapter_flag_exits_2(
        self, tmp_path: Path
    ) -> None:
        doc = _scaffold_flat_doc(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "prompt",
                str(doc),
                "hello",
                "--adapter",
                "knowledge",
            ],
        )
        assert result.exit_code == 2
        text = _joined_output(result)
        assert "only valid on multi-adapter" in text


class TestUnknownAdapterRejected:
    def test_adapter_not_in_declared_set_exits_2(self, tmp_path: Path) -> None:
        doc = _scaffold_multi_doc(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "prompt",
                str(doc),
                "hello",
                "--adapter",
                "ghost",
            ],
        )
        assert result.exit_code == 2
        text = _joined_output(result)
        assert "not declared" in text
        assert "'knowledge'" in text
        assert "'tone'" in text
