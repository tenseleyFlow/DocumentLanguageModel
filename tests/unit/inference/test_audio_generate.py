"""Audio inference helpers — prompt shaping, waveform loading, generation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from dlm.inference.audio_generate import format_audio_prompt, generate_audio, load_audios


class TestFormatAudioPrompt:
    def test_respects_user_placed_audio_token(self) -> None:
        prompt = "Please compare <audio> and explain."
        assert format_audio_prompt(prompt, audio_token="<audio>", num_audios=2) == prompt

    def test_prepends_one_token_per_audio(self) -> None:
        assert (
            format_audio_prompt("describe", audio_token="<audio>", num_audios=2)
            == "<audio><audio>\ndescribe"
        )

    def test_empty_prompt_emits_tokens_only(self) -> None:
        assert (
            format_audio_prompt("", audio_token="<audio>", num_audios=3) == "<audio><audio><audio>"
        )


class TestLoadAudios:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="audio not found"):
            load_audios([tmp_path / "missing.wav"], target_sample_rate=16_000)

    def test_downmixes_stereo_to_mono(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "stereo.wav"
        path.write_bytes(b"stub")

        fake_sf = ModuleType("soundfile")
        fake_sf.read = lambda _path, dtype, always_2d: (
            np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32),
            16_000,
        )
        monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

        [waveform] = load_audios([path], target_sample_rate=16_000)
        assert waveform.dtype == np.float32
        assert waveform.tolist() == pytest.approx([2.0, 6.0])

    def test_sample_rate_mismatch_refused_without_auto_resample(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "native.wav"
        path.write_bytes(b"stub")

        fake_sf = ModuleType("soundfile")
        fake_sf.read = lambda _path, dtype, always_2d: (np.array([1.0], dtype=np.float32), 22_050)
        monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

        with pytest.raises(ValueError, match="does not match pinned 16000 Hz"):
            load_audios([path], target_sample_rate=16_000, auto_resample=False)

    def test_sample_rate_mismatch_resamples_when_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "native.wav"
        path.write_bytes(b"stub")

        fake_sf = ModuleType("soundfile")
        fake_sf.read = lambda _path, dtype, always_2d: (
            np.array([1.0, 2.0], dtype=np.float32),
            22_050,
        )
        monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
        monkeypatch.setattr(
            "dlm.data.audio_resample.resample",
            lambda mono, src_sr, dst_sr: np.array([9.0, 8.0], dtype=np.float32),
        )

        [waveform] = load_audios([path], target_sample_rate=16_000, auto_resample=True)
        assert waveform.tolist() == pytest.approx([9.0, 8.0])


class _Inputs(dict[str, torch.Tensor]):
    def to(self, device: object) -> _Inputs:
        return self


class TestGenerateAudio:
    def test_generate_audio_decodes_response_only_tokens(self) -> None:
        class _Tokenizer:
            pad_token_id = 99

            def decode(self, tokens: torch.Tensor, skip_special_tokens: bool = True) -> str:
                assert tokens.tolist() == [4, 5]
                return "transcript"

        class _Processor:
            def __init__(self) -> None:
                self.tokenizer = _Tokenizer()

            def __call__(
                self,
                *,
                audios: list[np.ndarray],
                text: str,
                sampling_rate: int,
                return_tensors: str,
            ) -> _Inputs:
                assert len(audios) == 1
                assert text == "<audio>\nwhat happened?"
                assert sampling_rate == 16_000
                return _Inputs({"input_ids": torch.tensor([[1, 2, 3]])})

        class _Model:
            device = torch.device("cpu")

            def generate(self, **kwargs: object) -> torch.Tensor:
                assert kwargs["pad_token_id"] == 99
                return torch.tensor([[1, 2, 3, 4, 5]])

        out = generate_audio(
            _Model(),
            _Processor(),
            "what happened?",
            [np.array([1.0], dtype=np.float32)],
            audio_token="<audio>",
            sample_rate=16_000,
            max_new_tokens=2,
            temperature=0.0,
        )
        assert out == "transcript"
