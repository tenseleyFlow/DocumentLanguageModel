"""`dlm prompt --audio` flag validation (Sprint 35.2).

Parallel to `test_prompt_image_flag.py`. Covers:

- Passing --audio to a text-base doc exits 2 with an informative message.
- Omitting --audio on an audio-base doc exits 2 with an actionable hint.
- Combining --image + --audio exits 2 (each targets a different modality).

All exits happen before any HF-model load so CLI-level tests cover them
without touching torch / transformers / soundfile decode paths.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def _joined_output(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(text.split())


def _scaffold_text_doc(tmp_path: Path) -> Path:
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


def _scaffold_audio_doc(tmp_path: Path) -> Path:
    """Scaffold a doc with the Qwen2-Audio base pinned.

    Gating only kicks in on train/export; init accepts the spec as long
    as `--i-accept-license` is supplied.
    """
    doc = tmp_path / "audio.dlm"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_path / "home"),
            "init",
            str(doc),
            "--base",
            "qwen2-audio-7b-instruct",
            "--i-accept-license",
        ],
    )
    assert result.exit_code == 0, result.output
    return doc


class TestTextBaseRefusesAudio:
    def test_text_base_with_audio_exits_2(self, tmp_path: Path) -> None:
        doc = _scaffold_text_doc(tmp_path)
        wav = tmp_path / "x.wav"
        wav.write_bytes(b"fake wav bytes")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "prompt",
                str(doc),
                "hello",
                "--audio",
                str(wav),
            ],
        )
        assert result.exit_code == 2, result.output
        text = _joined_output(result)
        assert "--audio is only valid with audio-language bases" in text


class TestAudioBaseRequiresAudio:
    def test_audio_base_without_audio_exits_2(self, tmp_path: Path) -> None:
        doc = _scaffold_audio_doc(tmp_path)
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
        assert "audio-language" in text
        assert "--audio" in text


class TestImageAndAudioMutuallyExclusive:
    def test_combining_image_and_audio_exits_2(self, tmp_path: Path) -> None:
        doc = _scaffold_text_doc(tmp_path)
        wav = tmp_path / "clip.wav"
        wav.write_bytes(b"fake")
        img = tmp_path / "frame.png"
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
                "--audio",
                str(wav),
            ],
        )
        assert result.exit_code == 2, result.output
        text = _joined_output(result)
        assert "--image and --audio cannot be combined" in text
