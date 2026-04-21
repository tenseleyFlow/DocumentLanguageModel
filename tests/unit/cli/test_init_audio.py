"""`dlm init --audio` scaffold (Sprint 35.2 T11).

Parallel to `test_init_multimodal.py`. Covers:

- Default base flips to qwen2-audio-7b-instruct when --audio fires.
- Scaffold declares dlm_version >= 11 and includes an `::audio::` fence.
- Text base + --audio exits 2 with a pointer to the audio base.
- --audio + --multimodal rejected (mutually exclusive modalities).
- --audio + --template rejected (no audio template in v1).
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def _joined(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(text.split())


class TestAudioScaffold:
    def test_default_base_flips_to_qwen2_audio(self, tmp_path: Path) -> None:
        doc = tmp_path / "audio.dlm"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "init",
                str(doc),
                "--audio",
                "--i-accept-license",
            ],
        )
        assert result.exit_code == 0, _joined(result)
        body = doc.read_text(encoding="utf-8")
        assert "base_model: qwen2-audio-7b-instruct" in body

    def test_schema_v11_or_later(self, tmp_path: Path) -> None:
        doc = tmp_path / "audio.dlm"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "init",
                str(doc),
                "--audio",
                "--i-accept-license",
            ],
        )
        assert result.exit_code == 0, _joined(result)
        body = doc.read_text(encoding="utf-8")
        assert "dlm_version: 11" in body

    def test_body_includes_audio_fence(self, tmp_path: Path) -> None:
        doc = tmp_path / "audio.dlm"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "init",
                str(doc),
                "--audio",
                "--i-accept-license",
            ],
        )
        assert result.exit_code == 0, _joined(result)
        body = doc.read_text(encoding="utf-8")
        assert "::audio path=" in body
        assert "transcript=" in body

    def test_scaffold_parses_as_valid_dlm(self, tmp_path: Path) -> None:
        doc = tmp_path / "audio.dlm"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "init",
                str(doc),
                "--audio",
                "--i-accept-license",
            ],
        )
        assert result.exit_code == 0, _joined(result)
        from dlm.doc.parser import parse_file
        from dlm.doc.sections import SectionType

        parsed = parse_file(doc)
        # `>= 11` so future schema bumps don't regress this probe.
        assert parsed.frontmatter.dlm_version >= 11
        assert parsed.frontmatter.base_model == "qwen2-audio-7b-instruct"
        types = [s.type for s in parsed.sections]
        assert SectionType.AUDIO in types


class TestAudioRefusals:
    def test_text_base_with_audio_exits_2(self, tmp_path: Path) -> None:
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
                "--audio",
            ],
        )
        assert result.exit_code == 2, _joined(result)
        text = _joined(result)
        assert "audio-language base" in text

    def test_audio_with_multimodal_exits_2(self, tmp_path: Path) -> None:
        doc = tmp_path / "clash.dlm"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "init",
                str(doc),
                "--audio",
                "--multimodal",
            ],
        )
        assert result.exit_code == 2, _joined(result)
        text = _joined(result)
        assert "mutually exclusive" in text

    def test_audio_with_template_exits_2(self, tmp_path: Path) -> None:
        doc = tmp_path / "clash.dlm"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "init",
                str(doc),
                "--audio",
                "--template",
                "knowledge-base",
            ],
        )
        assert result.exit_code == 2, _joined(result)
        text = _joined(result)
        assert "mutually exclusive" in text


class TestNonAudioUnchanged:
    """Text init + VL init are unchanged after --audio lands."""

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
        assert "::audio" not in body
        assert "::image" not in body

    def test_multimodal_unaffected(self, tmp_path: Path) -> None:
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
        assert "::audio" not in body
        assert "::image path=" in body
