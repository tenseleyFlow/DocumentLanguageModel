"""`dlm init --multimodal` scaffold (Sprint 35 v1).

Covers:

- Default base flips to paligemma-3b-mix-224 when --multimodal fires.
- Explicit VL base still works.
- Text base + --multimodal exits 2 with a pointer to the VL bases.
- Scaffold declares dlm_version 10 and includes an `::image::` fence.
- --multimodal + --template rejected (templates don't carry VL bases
  in v1).
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def _joined(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(text.split())


class TestMultimodalScaffold:
    def test_default_base_flips_to_paligemma(self, tmp_path: Path) -> None:
        doc = tmp_path / "vl.dlm"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "init",
                str(doc),
                "--multimodal",
                "--i-accept-license",
            ],
        )
        assert result.exit_code == 0, _joined(result)
        body = doc.read_text(encoding="utf-8")
        assert "base_model: paligemma-3b-mix-224" in body

    def test_schema_v10_declared(self, tmp_path: Path) -> None:
        doc = tmp_path / "vl.dlm"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "init",
                str(doc),
                "--multimodal",
                "--i-accept-license",
            ],
        )
        assert result.exit_code == 0, _joined(result)
        body = doc.read_text(encoding="utf-8")
        assert "dlm_version: 10" in body

    def test_body_includes_image_fence(self, tmp_path: Path) -> None:
        doc = tmp_path / "vl.dlm"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "init",
                str(doc),
                "--multimodal",
                "--i-accept-license",
            ],
        )
        assert result.exit_code == 0, _joined(result)
        body = doc.read_text(encoding="utf-8")
        assert "::image path=" in body
        assert "alt=" in body

    def test_scaffold_parses_as_valid_dlm(self, tmp_path: Path) -> None:
        doc = tmp_path / "vl.dlm"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "init",
                str(doc),
                "--multimodal",
                "--i-accept-license",
            ],
        )
        assert result.exit_code == 0, _joined(result)
        # The scaffold must round-trip through our parser — otherwise
        # users get a cryptic error on their first `dlm show`.
        from dlm.doc.parser import parse_file
        from dlm.doc.sections import SectionType

        parsed = parse_file(doc)
        assert parsed.frontmatter.dlm_version == 10
        assert parsed.frontmatter.base_model == "paligemma-3b-mix-224"
        types = [s.type for s in parsed.sections]
        assert SectionType.IMAGE in types


class TestMultimodalRefusals:
    def test_text_base_with_multimodal_exits_2(self, tmp_path: Path) -> None:
        doc = tmp_path / "mismatch.dlm"
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
                "--multimodal",
            ],
        )
        assert result.exit_code == 2, _joined(result)
        text = _joined(result)
        assert "vision-language base" in text

    def test_multimodal_with_template_exits_2(self, tmp_path: Path) -> None:
        doc = tmp_path / "clash.dlm"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "init",
                str(doc),
                "--multimodal",
                "--template",
                "knowledge-base",
            ],
        )
        assert result.exit_code == 2, _joined(result)
        text = _joined(result)
        assert "mutually exclusive" in text


class TestNonMultimodalUnchanged:
    def test_default_init_still_text_scaffold(self, tmp_path: Path) -> None:
        doc = tmp_path / "text.dlm"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "init",
                str(doc),
            ],
        )
        assert result.exit_code == 0, _joined(result)
        body = doc.read_text(encoding="utf-8")
        assert "::image" not in body
        assert "dlm_version: 1" in body
