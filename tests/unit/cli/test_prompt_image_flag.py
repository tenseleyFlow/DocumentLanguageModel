"""`dlm prompt --image` flag validation (Sprint 35 v1).

- Passing --image to a text-base doc exits 2 with an informative message.
- Omitting --image on a VL-base doc exits 2 with an actionable hint.
- Both exits happen before any HF-model load, so CLI-level tests cover
  them without touching torch / transformers weights.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def _joined_output(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(text.split())


def _scaffold_text_doc(tmp_path: Path) -> Path:
    """Scaffold a flat text-base doc."""
    doc = tmp_path / "text.dlm"
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


def _scaffold_vl_doc(tmp_path: Path) -> Path:
    """Scaffold a doc with the PaliGemma base pinned.

    Gemma acceptance is a concern for `dlm train` / `dlm export`, not
    `dlm prompt` — the resolver only enforces acceptance if the store
    hasn't recorded one. We side-step by pinning base_model without
    triggering a real HF download (which the CLI doesn't do).
    """
    doc = tmp_path / "vl.dlm"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_path / "home"),
            "init",
            str(doc),
            "--base",
            "paligemma-3b-mix-224",
            "--i-accept-license",
        ],
    )
    assert result.exit_code == 0, result.output
    return doc


class TestTextBaseRefusesImage:
    def test_text_base_with_image_exits_2(self, tmp_path: Path) -> None:
        doc = _scaffold_text_doc(tmp_path)
        img = tmp_path / "x.png"
        img.write_bytes(b"\x89PNG fake")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "prompt",
                str(doc),
                "hello",
                "--image",
                str(img),
            ],
        )
        assert result.exit_code == 2, result.output
        text = _joined_output(result)
        assert "--image is only valid with vision-language bases" in text


class TestVlBaseRequiresImage:
    def test_vl_base_without_image_exits_2(self, tmp_path: Path) -> None:
        doc = _scaffold_vl_doc(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "prompt",
                str(doc),
                "hello",
            ],
        )
        assert result.exit_code == 2, result.output
        text = _joined_output(result)
        assert "vision-language" in text
        assert "--image" in text
