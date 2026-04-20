"""`dlm export --adapter` flag validation (Sprint 20b)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def _joined(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(text.split())


def _init_flat(tmp_path: Path) -> Path:
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


def _init_multi(tmp_path: Path) -> Path:
    doc = _init_flat(tmp_path)
    original = doc.read_text(encoding="utf-8")
    fm_end = original.find("\n---\n", original.find("---") + 3)
    new_fm = (
        original[:fm_end]
        + "\ntraining:\n  adapters:\n    knowledge: {}\n    tone: {}\n"
        + original[fm_end:]
    )
    doc.write_text(new_fm, encoding="utf-8")
    return doc


class TestExportAdapterFlagValidation:
    def test_flat_doc_rejects_adapter_flag(self, tmp_path: Path) -> None:
        doc = _init_flat(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "export",
                str(doc),
                "--adapter",
                "knowledge",
                "--skip-ollama",
            ],
        )
        assert result.exit_code == 2
        assert "only valid on multi-adapter" in _joined(result)

    def test_unknown_adapter_name_rejected(self, tmp_path: Path) -> None:
        doc = _init_multi(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "export",
                str(doc),
                "--adapter",
                "ghost",
                "--skip-ollama",
            ],
        )
        assert result.exit_code == 2
        text = _joined(result)
        assert "not declared" in text
        assert "'knowledge'" in text
        assert "'tone'" in text


class TestExportAdapterMixValidation:
    def test_flag_mutex_with_adapter(self, tmp_path: Path) -> None:
        doc = _init_multi(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "export",
                str(doc),
                "--adapter",
                "knowledge",
                "--adapter-mix",
                "knowledge:1.0,tone:0.5",
                "--skip-ollama",
            ],
        )
        assert result.exit_code == 2
        assert "mutually exclusive" in _joined(result)

    def test_flat_doc_rejects_adapter_mix(self, tmp_path: Path) -> None:
        doc = _init_flat(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "export",
                str(doc),
                "--adapter-mix",
                "knowledge:1.0",
                "--skip-ollama",
            ],
        )
        assert result.exit_code == 2
        assert "only valid on multi-adapter" in _joined(result)

    def test_malformed_mix_rejected(self, tmp_path: Path) -> None:
        doc = _init_multi(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "export",
                str(doc),
                "--adapter-mix",
                "knowledge",  # missing weight
                "--skip-ollama",
            ],
        )
        assert result.exit_code == 2
        assert "missing a weight" in _joined(result)

    def test_unknown_name_in_mix_rejected(self, tmp_path: Path) -> None:
        doc = _init_multi(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "export",
                str(doc),
                "--adapter-mix",
                "ghost:1.0",
                "--skip-ollama",
            ],
        )
        assert result.exit_code == 2
        text = _joined(result)
        assert "not declared" in text
        assert "ghost" in text
