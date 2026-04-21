"""Audio preprocessor — cache hit/miss, sample-rate refusal, truncation.

Uses soundfile to write a real .wav fixture (so the preprocessor's
native-rate refusal + truncation branches actually execute) and a
stub HF processor to avoid the heavy AutoProcessor import. Wiring
into a real Qwen2-Audio processor is exercised in the slow audio
integration test (T12).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlm.data.audio_cache import AudioCache
from dlm.data.audio_preprocessor import AudioSampleRateMismatch, preprocess_audio


class _StubProcessor:
    """Deterministic fake audio processor.

    `feature_extractor.sampling_rate` drives the fingerprint;
    `calls` counts invocations so cache-hit tests can prove the
    processor didn't run. The `__call__` records the number of input
    samples it received so the truncation assertions have a handle.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.last_num_samples: int | None = None

        class _FE:
            sampling_rate = 16_000
            feature_size = 80
            n_fft = 400
            hop_length = 160
            chunk_length = 30
            padding_value = 0.0
            return_attention_mask = True

        self.feature_extractor = _FE()

    def __call__(
        self,
        *,
        audios: np.ndarray,
        sampling_rate: int,
        return_tensors: str,
    ) -> dict[str, np.ndarray]:
        self.calls += 1
        _ = sampling_rate
        _ = return_tensors
        self.last_num_samples = int(audios.shape[-1])
        return {
            "input_features": np.full(
                (1, 80, 3000),
                float(self.calls),
                dtype=np.float32,
            ),
        }


def _write_wav(path: Path, *, sample_rate: int, seconds: float) -> None:
    """Write a mono float32 sine wave of the given length at `sample_rate`."""
    import soundfile as sf

    num_samples = int(round(seconds * sample_rate))
    t = np.linspace(0.0, seconds, num_samples, dtype=np.float32)
    data = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    sf.write(str(path), data, sample_rate, subtype="FLOAT")


@pytest.fixture
def tiny_wav_16k(tmp_path: Path) -> Path:
    path = tmp_path / "clip-16k.wav"
    _write_wav(path, sample_rate=16_000, seconds=0.5)
    return path


@pytest.fixture
def tiny_wav_48k(tmp_path: Path) -> Path:
    path = tmp_path / "clip-48k.wav"
    _write_wav(path, sample_rate=48_000, seconds=0.5)
    return path


class TestPreprocessAudioNoCache:
    def test_runs_processor(self, tiny_wav_16k: Path) -> None:
        proc = _StubProcessor()
        result = preprocess_audio(
            blob_path=tiny_wav_16k,
            blob_sha="a" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            cache=None,
        )
        assert proc.calls == 1
        assert result.cache_hit is False
        assert result.input_features.shape == (1, 80, 3000)
        assert result.input_features.dtype == np.float32

    def test_sample_rate_mismatch_refused(self, tiny_wav_48k: Path) -> None:
        proc = _StubProcessor()
        with pytest.raises(AudioSampleRateMismatch, match="48000"):
            preprocess_audio(
                blob_path=tiny_wav_48k,
                blob_sha="a" * 64,
                processor=proc,
                sample_rate=16_000,
                max_length_seconds=30.0,
                cache=None,
            )
        assert proc.calls == 0

    def test_sample_rate_mismatch_hint_mentions_auto_resample(self, tiny_wav_48k: Path) -> None:
        """Error guides the user to the opt-in flag."""
        proc = _StubProcessor()
        with pytest.raises(AudioSampleRateMismatch, match="auto_resample"):
            preprocess_audio(
                blob_path=tiny_wav_48k,
                blob_sha="a" * 64,
                processor=proc,
                sample_rate=16_000,
                max_length_seconds=30.0,
                cache=None,
            )

    def test_auto_resample_routes_through_resampler(
        self, tiny_wav_48k: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """auto_resample=True on a mismatch calls dlm.data.audio_resample.resample."""
        calls: list[tuple[int, int]] = []

        def fake_resample(waveform: np.ndarray, *, src_sr: int, dst_sr: int) -> np.ndarray:
            calls.append((src_sr, dst_sr))
            out_len = int(waveform.shape[0] * dst_sr / src_sr)
            return np.zeros(out_len, dtype=np.float32)

        from dlm.data import audio_resample as rs_mod

        monkeypatch.setattr(rs_mod, "resample", fake_resample)

        proc = _StubProcessor()
        result = preprocess_audio(
            blob_path=tiny_wav_48k,
            blob_sha="a" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            cache=None,
            auto_resample=True,
        )
        assert calls == [(48_000, 16_000)]
        assert proc.calls == 1
        assert result.cache_hit is False

    def test_truncates_to_max_length(self, tmp_path: Path) -> None:
        # 2-second clip, cap at 0.5 s → processor should see 8000 samples.
        path = tmp_path / "long.wav"
        _write_wav(path, sample_rate=16_000, seconds=2.0)
        proc = _StubProcessor()
        preprocess_audio(
            blob_path=path,
            blob_sha="d" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=0.5,
            cache=None,
        )
        assert proc.last_num_samples == 8_000

    def test_short_clip_not_padded_by_preprocessor(self, tiny_wav_16k: Path) -> None:
        # The preprocessor's job is truncate-only; any padding belongs
        # to the HF feature extractor. Assert we didn't silently pad.
        proc = _StubProcessor()
        preprocess_audio(
            blob_path=tiny_wav_16k,
            blob_sha="e" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            cache=None,
        )
        assert proc.last_num_samples == 8_000  # 0.5 s × 16 kHz


class TestPreprocessAudioWithCache:
    def test_first_call_misses_then_writes(
        self,
        tiny_wav_16k: Path,
        tmp_path: Path,
    ) -> None:
        proc = _StubProcessor()
        cache = AudioCache(tmp_path / "audio")
        result = preprocess_audio(
            blob_path=tiny_wav_16k,
            blob_sha="a" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            cache=cache,
        )
        assert result.cache_hit is False
        assert proc.calls == 1
        assert any(cache.root.rglob("*.npz"))

    def test_second_call_hits_cache(
        self,
        tiny_wav_16k: Path,
        tmp_path: Path,
    ) -> None:
        proc = _StubProcessor()
        cache = AudioCache(tmp_path / "audio")
        first = preprocess_audio(
            blob_path=tiny_wav_16k,
            blob_sha="a" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            cache=cache,
        )
        second = preprocess_audio(
            blob_path=tiny_wav_16k,
            blob_sha="a" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            cache=cache,
        )
        assert second.cache_hit is True
        assert proc.calls == 1
        np.testing.assert_array_equal(first.input_features, second.input_features)

    def test_different_max_length_misses(
        self,
        tiny_wav_16k: Path,
        tmp_path: Path,
    ) -> None:
        proc = _StubProcessor()
        cache = AudioCache(tmp_path / "audio")
        preprocess_audio(
            blob_path=tiny_wav_16k,
            blob_sha="a" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            cache=cache,
        )
        preprocess_audio(
            blob_path=tiny_wav_16k,
            blob_sha="a" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=60.0,
            cache=cache,
        )
        assert proc.calls == 2

    def test_different_blob_sha_misses(
        self,
        tiny_wav_16k: Path,
        tmp_path: Path,
    ) -> None:
        proc = _StubProcessor()
        cache = AudioCache(tmp_path / "audio")
        preprocess_audio(
            blob_path=tiny_wav_16k,
            blob_sha="a" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            cache=cache,
        )
        preprocess_audio(
            blob_path=tiny_wav_16k,
            blob_sha="b" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            cache=cache,
        )
        assert proc.calls == 2

    def test_auto_resample_key_disjoint_from_native_key(
        self,
        tiny_wav_16k: Path,
        tmp_path: Path,
    ) -> None:
        """auto_resample=True vs =False write to separate cache entries.

        Guards the correctness invariant: an entry cached without
        resampling must not satisfy a later call that asked to resample
        (and vice versa) — the processor input was different.
        """
        proc = _StubProcessor()
        cache = AudioCache(tmp_path / "audio")
        preprocess_audio(
            blob_path=tiny_wav_16k,
            blob_sha="a" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            cache=cache,
            auto_resample=False,
        )
        preprocess_audio(
            blob_path=tiny_wav_16k,
            blob_sha="a" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            cache=cache,
            auto_resample=True,
        )
        # Two separate cache entries → processor ran twice.
        assert proc.calls == 2
        entries = list(cache.root.rglob("*.npz"))
        assert len(entries) == 2


class TestPreprocessAudioMonoMix:
    def test_stereo_averaged(self, tmp_path: Path) -> None:
        import soundfile as sf

        path = tmp_path / "stereo.wav"
        num_samples = 8_000  # 0.5 s @ 16 kHz
        # L channel = +1, R channel = -1 → mean is 0 across samples.
        stereo = np.stack(
            [
                np.ones(num_samples, dtype=np.float32),
                -np.ones(num_samples, dtype=np.float32),
            ],
            axis=1,
        )
        sf.write(str(path), stereo, 16_000, subtype="FLOAT")

        proc = _StubProcessor()
        preprocess_audio(
            blob_path=path,
            blob_sha="f" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            cache=None,
        )
        # The stub records the samples it saw; mean of ±1 channels → 0s.
        assert proc.last_num_samples == num_samples


class TestPreprocessAudioReturnsNumpy:
    def test_coerces_non_ndarray(self, tiny_wav_16k: Path) -> None:
        class WrappedProc(_StubProcessor):
            def __call__(
                self,
                *,
                audios: np.ndarray,
                sampling_rate: int,
                return_tensors: str,
            ) -> dict[str, list]:  # type: ignore[override]
                self.calls += 1
                _ = audios
                _ = sampling_rate
                _ = return_tensors
                return {"input_features": [[[1.0, 2.0], [3.0, 4.0]]]}

        proc = WrappedProc()
        result = preprocess_audio(
            blob_path=tiny_wav_16k,
            blob_sha="c" * 64,
            processor=proc,
            sample_rate=16_000,
            max_length_seconds=30.0,
            cache=None,
        )
        assert isinstance(result.input_features, np.ndarray)
        assert result.input_features.dtype == np.float32


class TestStorePathAudioCacheDir:
    """StorePath surfaces `audio_cache_dir` the same way as `vl_cache_dir`."""

    def test_audio_cache_dir_path(self, tmp_path: Path) -> None:
        from dlm.store.paths import StorePath

        sp = StorePath(root=tmp_path / "store")
        assert sp.audio_cache_dir == tmp_path / "store" / "audio-cache"
        # Lazy — does not exist until something writes.
        assert not sp.audio_cache_dir.exists()

    def test_audio_cache_dir_not_same_as_vl(self, tmp_path: Path) -> None:
        from dlm.store.paths import StorePath

        sp = StorePath(root=tmp_path / "store")
        assert sp.audio_cache_dir != sp.vl_cache_dir
