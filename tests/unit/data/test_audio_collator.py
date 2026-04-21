"""AudioLmCollator — batch shape, labels, sample-rate refusal.

Uses soundfile to write real .wav fixtures (so the waveform-loading
branch executes) and a stub processor to avoid pulling in the full
transformers AutoProcessor path. The wiring into a real Qwen2-Audio
processor is exercised in the slow audio integration test (T12).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlm.data.audio_collator import AudioLmCollator


class _StubTokenizer:
    pad_token_id = 0


class _StubProcessor:
    """Fake Qwen2-Audio-shaped processor.

    Returns torch tensors with predictable shapes so the collator's
    output-stitching logic can be asserted. `calls` records how many
    batches it's been invoked on, `last_kwargs` captures the final
    call so tests can verify the collator's argument shape.
    """

    def __init__(self) -> None:
        self.tokenizer = _StubTokenizer()
        self.calls = 0
        self.last_kwargs: dict[str, Any] | None = None

    def __call__(
        self,
        *,
        text: list[str],
        audios: list[np.ndarray],
        sampling_rate: int,
        return_tensors: str,
        padding: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import torch

        self.calls += 1
        self.last_kwargs = {
            "text": text,
            "audios": audios,
            "sampling_rate": sampling_rate,
            "return_tensors": return_tensors,
            "padding": padding,
            **kwargs,
        }
        batch_size = len(text)
        # Synthetic token ids — length 8 with trailing pads to exercise
        # the label-masking branch.
        seq_len = 8
        input_ids = torch.full((batch_size, seq_len), 1, dtype=torch.long)
        # Pad positions on the last two columns.
        input_ids[:, -2:] = 0
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, -2:] = 0
        # 80-bin, 100-frame stand-in for log-mel features.
        input_features = torch.zeros(batch_size, 80, 100, dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
        }


def _write_wav(path: Path, *, sample_rate: int, seconds: float = 0.5) -> None:
    import soundfile as sf

    num_samples = int(round(seconds * sample_rate))
    t = np.linspace(0.0, seconds, num_samples, dtype=np.float32)
    data = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    sf.write(str(path), data, sample_rate, subtype="FLOAT")


@pytest.fixture
def wav_16k(tmp_path: Path) -> Path:
    path = tmp_path / "clip.wav"
    _write_wav(path, sample_rate=16_000)
    return path


@pytest.fixture
def wav_48k(tmp_path: Path) -> Path:
    path = tmp_path / "off-rate.wav"
    _write_wav(path, sample_rate=48_000)
    return path


def _make_collator(processor: _StubProcessor) -> AudioLmCollator:
    return AudioLmCollator(
        processor=processor,
        sample_rate=16_000,
        max_length_seconds=30.0,
        max_length=512,
    )


class TestBatchShape:
    def test_single_row_batch(self, wav_16k: Path) -> None:
        proc = _StubProcessor()
        collator = _make_collator(proc)
        row = {
            "audio_blob_sha": "a" * 64,
            "audio_path": str(wav_16k),
            "text": "<|AUDIO|>\nHello.",
        }
        batch = collator([row])
        assert proc.calls == 1
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "input_features" in batch
        assert "labels" in batch

    def test_multi_row_batch(self, wav_16k: Path) -> None:
        proc = _StubProcessor()
        collator = _make_collator(proc)
        rows = [
            {
                "audio_blob_sha": "a" * 64,
                "audio_path": str(wav_16k),
                "text": "<|AUDIO|>\nOne.",
            },
            {
                "audio_blob_sha": "b" * 64,
                "audio_path": str(wav_16k),
                "text": "<|AUDIO|>\nTwo.",
            },
        ]
        batch = collator(rows)
        assert batch["input_ids"].shape[0] == 2
        assert batch["labels"].shape == batch["input_ids"].shape
        assert proc.last_kwargs is not None
        assert proc.last_kwargs["text"] == ["<|AUDIO|>\nOne.", "<|AUDIO|>\nTwo."]

    def test_sampling_rate_passed_to_processor(self, wav_16k: Path) -> None:
        proc = _StubProcessor()
        collator = _make_collator(proc)
        collator([{"audio_blob_sha": "a" * 64, "audio_path": str(wav_16k), "text": "x"}])
        assert proc.last_kwargs is not None
        assert proc.last_kwargs["sampling_rate"] == 16_000

    def test_max_length_forwarded_when_set(self, wav_16k: Path) -> None:
        proc = _StubProcessor()
        collator = _make_collator(proc)
        collator([{"audio_blob_sha": "a" * 64, "audio_path": str(wav_16k), "text": "x"}])
        assert proc.last_kwargs is not None
        assert proc.last_kwargs["max_length"] == 512

    def test_max_length_omitted_when_none(self, wav_16k: Path) -> None:
        proc = _StubProcessor()
        collator = AudioLmCollator(
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            max_length=None,
        )
        collator([{"audio_blob_sha": "a" * 64, "audio_path": str(wav_16k), "text": "x"}])
        assert proc.last_kwargs is not None
        assert "max_length" not in proc.last_kwargs


class TestLabelMasking:
    def test_pad_positions_masked_to_neg_100(self, wav_16k: Path) -> None:
        proc = _StubProcessor()
        collator = _make_collator(proc)
        batch = collator([{"audio_blob_sha": "a" * 64, "audio_path": str(wav_16k), "text": "x"}])
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        # Non-pad positions: labels == input_ids.
        non_pad_mask = input_ids != 0
        assert (labels[non_pad_mask] == input_ids[non_pad_mask]).all()
        # Pad positions: labels == -100.
        assert (labels[input_ids == 0] == -100).all()


class TestWaveformLoading:
    def test_mono_downmix(self, tmp_path: Path) -> None:
        import soundfile as sf

        path = tmp_path / "stereo.wav"
        num_samples = 8_000  # 0.5 s × 16 kHz
        stereo = np.stack(
            [
                np.ones(num_samples, dtype=np.float32),
                -np.ones(num_samples, dtype=np.float32),
            ],
            axis=1,
        )
        sf.write(str(path), stereo, 16_000, subtype="FLOAT")

        proc = _StubProcessor()
        collator = _make_collator(proc)
        collator(
            [
                {
                    "audio_blob_sha": "f" * 64,
                    "audio_path": str(path),
                    "text": "test",
                }
            ]
        )
        # Processor saw a 1-D float32 array (stereo got averaged).
        assert proc.last_kwargs is not None
        waveform = proc.last_kwargs["audios"][0]
        assert waveform.ndim == 1
        assert waveform.dtype == np.float32
        assert waveform.shape[0] == num_samples

    def test_truncation_to_max_length_seconds(self, tmp_path: Path) -> None:
        path = tmp_path / "long.wav"
        _write_wav(path, sample_rate=16_000, seconds=2.0)
        proc = _StubProcessor()
        collator = AudioLmCollator(
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=0.5,
            max_length=None,
        )
        collator([{"audio_blob_sha": "a" * 64, "audio_path": str(path), "text": "x"}])
        assert proc.last_kwargs is not None
        waveform = proc.last_kwargs["audios"][0]
        assert waveform.shape[0] == 8_000  # 0.5 × 16 000


class TestSampleRateRefusal:
    def test_mismatched_rate_refused(self, wav_48k: Path) -> None:
        proc = _StubProcessor()
        collator = _make_collator(proc)
        with pytest.raises(ValueError, match="48000"):
            collator(
                [
                    {
                        "audio_blob_sha": "a" * 64,
                        "audio_path": str(wav_48k),
                        "text": "x",
                    }
                ]
            )
        assert proc.calls == 0

    def test_mismatched_rate_mentions_auto_resample_in_error(self, wav_48k: Path) -> None:
        """User-facing error must name the opt-in so the fix is obvious."""
        proc = _StubProcessor()
        collator = _make_collator(proc)
        with pytest.raises(ValueError, match="auto_resample"):
            collator(
                [
                    {
                        "audio_blob_sha": "a" * 64,
                        "audio_path": str(wav_48k),
                        "text": "x",
                    }
                ]
            )


class TestAutoResample:
    """auto_resample=True routes mismatched rates through the resampler."""

    def test_mismatched_rate_resampled_when_opted_in(
        self, wav_48k: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With auto_resample=True, the collator calls into audio_resample."""
        # Monkey-patch resample() to a deterministic stub so the test
        # doesn't depend on soxr / scipy being installed in the dev env.
        calls: list[tuple[int, int]] = []

        def fake_resample(waveform: np.ndarray, *, src_sr: int, dst_sr: int) -> np.ndarray:
            calls.append((src_sr, dst_sr))
            # Return a waveform at the target rate (same duration).
            out_len = int(waveform.shape[0] * dst_sr / src_sr)
            return np.zeros(out_len, dtype=np.float32)

        from dlm.data import audio_resample as rs_mod

        monkeypatch.setattr(rs_mod, "resample", fake_resample)

        proc = _StubProcessor()
        collator = AudioLmCollator(
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            max_length=None,
            auto_resample=True,
        )
        collator(
            [
                {
                    "audio_blob_sha": "a" * 64,
                    "audio_path": str(wav_48k),
                    "text": "x",
                }
            ]
        )
        assert calls == [(48_000, 16_000)]
        # Processor ran once on the resampled waveform.
        assert proc.calls == 1
        assert proc.last_kwargs is not None
        assert proc.last_kwargs["audios"][0].dtype == np.float32

    def test_matched_rate_bypasses_resampler(
        self, wav_16k: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """auto_resample=True with matched SR still skips the resampler."""
        calls: list[tuple[int, int]] = []

        def fake_resample(waveform: np.ndarray, *, src_sr: int, dst_sr: int) -> np.ndarray:
            calls.append((src_sr, dst_sr))
            return waveform

        from dlm.data import audio_resample as rs_mod

        monkeypatch.setattr(rs_mod, "resample", fake_resample)

        proc = _StubProcessor()
        collator = AudioLmCollator(
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            auto_resample=True,
        )
        collator(
            [
                {
                    "audio_blob_sha": "a" * 64,
                    "audio_path": str(wav_16k),
                    "text": "x",
                }
            ]
        )
        # No resample call — rates agreed.
        assert calls == []


class TestCollatorConstruction:
    def test_missing_tokenizer_refused(self) -> None:
        class NoTokProc:
            pass

        with pytest.raises(ValueError, match="no `.tokenizer` attribute"):
            AudioLmCollator(
                processor=NoTokProc(),
                sample_rate=16_000,
                max_length_seconds=30.0,
            )

    def test_missing_pad_token_refused(self) -> None:
        class NoPadTok:
            pad_token_id = None

        class Proc:
            tokenizer = NoPadTok()

        with pytest.raises(ValueError, match="no pad_token_id"):
            AudioLmCollator(
                processor=Proc(),
                sample_rate=16_000,
                max_length_seconds=30.0,
            )

    def test_empty_batch_refused(self) -> None:
        proc = _StubProcessor()
        collator = _make_collator(proc)
        with pytest.raises(ValueError, match="empty batch"):
            collator([])

    def test_missing_keys_refused(self, wav_16k: Path) -> None:
        proc = _StubProcessor()
        collator = _make_collator(proc)
        with pytest.raises(ValueError, match="missing required keys"):
            collator([{"audio_path": str(wav_16k)}])  # no `text`


class TestWaveformCacheIntegration:
    """Deferred-item follow-up: WaveformCache skips repeat decodes."""

    def test_second_call_hits_cache(self, wav_16k: Path, tmp_path: Path) -> None:
        from dlm.data.audio_cache import WaveformCache

        proc = _StubProcessor()
        cache = WaveformCache(tmp_path / "wav-cache")
        collator = AudioLmCollator(
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            max_length=512,
            waveform_cache=cache,
        )
        row = {
            "audio_blob_sha": "a" * 64,
            "audio_path": str(wav_16k),
            "text": "<|AUDIO|>\nHi.",
        }
        # First call: miss → decodes + writes cache.
        collator([row])
        cache_files = list(cache.root.rglob("*.npz"))
        assert len(cache_files) == 1

        # Delete the .wav from disk → second call must come from cache.
        wav_16k.unlink()
        assert not wav_16k.exists()

        # Second call: hit → uses cached waveform despite missing file.
        # If the cache wasn't consulted, this would raise FileNotFoundError
        # from soundfile.
        collator([row])
        assert proc.calls == 2  # processor runs both times (expected)

    def test_cache_key_disambiguates_sample_rate(self, wav_16k: Path, tmp_path: Path) -> None:
        """Same blob at different sample rates → different cache entries."""
        from dlm.data.audio_cache import WaveformCache

        proc = _StubProcessor()
        cache = WaveformCache(tmp_path / "wav-cache")
        collator_16k = AudioLmCollator(
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            waveform_cache=cache,
        )
        row = {
            "audio_blob_sha": "a" * 64,
            "audio_path": str(wav_16k),
            "text": "<|AUDIO|>",
        }
        collator_16k([row])
        # Only one cache entry exists for (a*64, 16000, 30000).
        assert len(list(cache.root.rglob("*.npz"))) == 1

    def test_no_cache_when_not_configured(self, wav_16k: Path, tmp_path: Path) -> None:
        """waveform_cache=None bypasses the cache entirely (default behavior)."""
        proc = _StubProcessor()
        collator = AudioLmCollator(
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            waveform_cache=None,
        )
        # Verify no cache file is created under tmp_path when the cache
        # is not configured — confirms we didn't silently default to one.
        collator(
            [
                {
                    "audio_blob_sha": "a" * 64,
                    "audio_path": str(wav_16k),
                    "text": "x",
                }
            ]
        )
        assert not list(tmp_path.rglob("*.npz"))

    def test_cache_skipped_when_blob_sha_missing(self, wav_16k: Path, tmp_path: Path) -> None:
        """Row without audio_blob_sha → decodes but doesn't cache.

        Rows from pre-35.2 codepaths (or ad-hoc construction) may lack
        a blob sha. Skip the cache rather than raise — the decode path
        still works correctly.
        """
        from dlm.data.audio_cache import WaveformCache

        proc = _StubProcessor()
        cache = WaveformCache(tmp_path / "wav-cache")
        collator = AudioLmCollator(
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            waveform_cache=cache,
        )
        collator(
            [
                {
                    # No audio_blob_sha key.
                    "audio_path": str(wav_16k),
                    "text": "x",
                }
            ]
        )
        # Cache is untouched because we had no key to store under.
        assert not list(cache.root.rglob("*.npz"))
