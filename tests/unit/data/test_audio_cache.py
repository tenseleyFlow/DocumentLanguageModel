"""Audio cache — key stability, atomic I/O, processor fingerprint.

Mirrors `test_vl_cache.py`. Covers:

- `AudioCacheKey.as_filename` / `shard` are deterministic.
- Different sample_rate / max_length_ms / processor_sha produce
  different filenames.
- Round-trip: put → get returns byte-identical array.
- Miss on empty store, miss on corrupt file.
- `processor_sha256` is stable across repeat calls + pinned on instance.
- Different feature-extractor constants drift the sha.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dlm.data.audio_cache import (
    AudioCache,
    AudioCacheKey,
    WaveformCache,
    WaveformCacheKey,
    processor_sha256,
)


def _key(**overrides: object) -> AudioCacheKey:
    defaults = {
        "blob_sha": "a" * 64,
        "processor_sha": "b" * 64,
        "sample_rate": 16_000,
        "max_length_ms": 30_000,
    }
    defaults.update(overrides)
    return AudioCacheKey(**defaults)  # type: ignore[arg-type]


class TestAudioCacheKey:
    def test_filename_shape(self) -> None:
        key = _key()
        assert key.as_filename() == f"{'a' * 64}.{'b' * 12}.16000.30000.npz"

    def test_shard_is_two_prefix(self) -> None:
        assert _key(blob_sha="cd" + "0" * 62).shard() == "cd"

    def test_different_sample_rate_different_filename(self) -> None:
        a = _key(sample_rate=16_000)
        b = _key(sample_rate=48_000)
        assert a.as_filename() != b.as_filename()

    def test_different_max_length_different_filename(self) -> None:
        a = _key(max_length_ms=30_000)
        b = _key(max_length_ms=60_000)
        assert a.as_filename() != b.as_filename()

    def test_different_processor_different_filename(self) -> None:
        a = _key(processor_sha="1" * 64)
        b = _key(processor_sha="2" * 64)
        assert a.as_filename() != b.as_filename()

    def test_key_is_frozen(self) -> None:
        key = _key()
        with pytest.raises(AttributeError):
            key.blob_sha = "x" * 64  # type: ignore[misc]

    def test_auto_resample_default_false_absent_from_filename(self) -> None:
        """Default False → filename stays v11-compatible (no `.rs` suffix).

        Guards backward-compat: an existing cache populated before the
        auto_resample field lands still hits on the same filename when
        the caller doesn't opt in.
        """
        assert ".rs" not in _key().as_filename()

    def test_auto_resample_true_adds_suffix(self) -> None:
        a = _key(auto_resample=False)
        b = _key(auto_resample=True)
        assert a.as_filename() != b.as_filename()
        assert ".rs" in b.as_filename()


class TestAudioCacheRoundTrip:
    def test_miss_on_empty(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path / "audio")
        assert cache.get(_key()) is None

    def test_put_then_get(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path / "audio")
        tensor = np.arange(80 * 3000, dtype=np.float32).reshape(1, 80, 3000)
        cache.put(_key(), tensor)
        loaded = cache.get(_key())
        assert loaded is not None
        np.testing.assert_array_equal(loaded, tensor)
        assert loaded.dtype == np.float32

    def test_put_creates_shard_dir(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path / "audio")
        key = _key(blob_sha="ef" + "0" * 62)
        cache.put(key, np.zeros((1,), dtype=np.float32))
        assert (tmp_path / "audio" / "ef").is_dir()

    def test_exists_flips_after_put(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path / "audio")
        key = _key()
        assert cache.exists(key) is False
        cache.put(key, np.zeros((1,), dtype=np.float32))
        assert cache.exists(key) is True

    def test_corrupt_file_treated_as_miss(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path / "audio")
        key = _key()
        cache.put(key, np.zeros((1,), dtype=np.float32))
        cache.path_for(key).write_bytes(b"not a real npz")
        assert cache.get(key) is None

    def test_clear_removes_tree(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path / "audio")
        cache.put(_key(), np.zeros((1,), dtype=np.float32))
        cache.clear()
        assert not (tmp_path / "audio").exists()


class TestProcessorSha256:
    def _make_processor(self, **attrs: object) -> SimpleNamespace:
        defaults: dict[str, object] = {
            "sampling_rate": 16_000,
            "feature_size": 80,
            "n_fft": 400,
            "hop_length": 160,
            "chunk_length": 30,
            "padding_value": 0.0,
            "return_attention_mask": True,
        }
        defaults.update(attrs)
        fe = SimpleNamespace(**defaults)
        return SimpleNamespace(feature_extractor=fe)

    def test_stable_across_calls(self) -> None:
        proc = self._make_processor()
        assert processor_sha256(proc) == processor_sha256(proc)

    def test_pinned_on_instance(self) -> None:
        proc = self._make_processor()
        first = processor_sha256(proc)
        # Mutate a field that would drift the sha if recomputed — the
        # pinned cache returns the original so repeat calls stay O(1).
        proc.feature_extractor.sampling_rate = 48_000
        assert processor_sha256(proc) == first

    def test_different_sample_rate_different_sha(self) -> None:
        a = self._make_processor(sampling_rate=16_000)
        b = self._make_processor(sampling_rate=48_000)
        assert processor_sha256(a) != processor_sha256(b)

    def test_different_n_fft_different_sha(self) -> None:
        a = self._make_processor(n_fft=400)
        b = self._make_processor(n_fft=1024)
        assert processor_sha256(a) != processor_sha256(b)

    def test_different_hop_length_different_sha(self) -> None:
        a = self._make_processor(hop_length=160)
        b = self._make_processor(hop_length=320)
        assert processor_sha256(a) != processor_sha256(b)

    def test_different_feature_extractor_class_different_sha(self) -> None:
        class FeA:
            sampling_rate = 16_000
            feature_size = 80
            n_fft = 400
            hop_length = 160

        class FeB:
            sampling_rate = 16_000
            feature_size = 80
            n_fft = 400
            hop_length = 160

        proc_a = SimpleNamespace(feature_extractor=FeA())
        proc_b = SimpleNamespace(feature_extractor=FeB())
        assert processor_sha256(proc_a) != processor_sha256(proc_b)

    def test_nested_feature_extractor_fields_are_readable(self) -> None:
        proc = SimpleNamespace(
            feature_extractor=SimpleNamespace(
                sampling_rate=16_000,
                feature_size=(80, 2),
                n_fft=400,
                hop_length=160,
                chunk_length={"seconds": 30},
                padding_value=0.0,
                return_attention_mask=True,
            )
        )
        sha = processor_sha256(proc)
        assert len(sha) == 64

    def test_exotic_feature_field_stringifies_stably(self) -> None:
        proc = SimpleNamespace(
            feature_extractor=SimpleNamespace(
                sampling_rate=16_000,
                feature_size=80,
                n_fft=400,
                hop_length=160,
                chunk_length=object(),
                padding_value=0.0,
                return_attention_mask=True,
            )
        )
        sha = processor_sha256(proc)
        assert len(sha) == 64


# --- WaveformCache (35.2 deferred-item follow-up) ---------------------------


def _wkey(**overrides: object) -> WaveformCacheKey:
    defaults: dict[str, object] = {
        "blob_sha": "a" * 64,
        "sample_rate": 16_000,
        "max_length_ms": 30_000,
    }
    defaults.update(overrides)
    return WaveformCacheKey(**defaults)  # type: ignore[arg-type]


class TestWaveformCacheKey:
    def test_filename_shape(self) -> None:
        k = _wkey()
        assert k.as_filename() == f"{'a' * 64}.16000.30000.wav.npz"

    def test_shard_is_two_prefix(self) -> None:
        assert _wkey(blob_sha="cd" + "0" * 62).shard() == "cd"

    def test_different_sample_rate_different_filename(self) -> None:
        assert _wkey(sample_rate=16_000).as_filename() != _wkey(sample_rate=48_000).as_filename()

    def test_different_max_length_different_filename(self) -> None:
        assert (
            _wkey(max_length_ms=30_000).as_filename() != _wkey(max_length_ms=60_000).as_filename()
        )

    def test_key_no_processor_sha(self) -> None:
        """Waveform cache is pre-processor; key should omit processor_sha."""
        # Distinct filenames from AudioCacheKey even for overlapping params
        # — the layout is intentionally separate.
        k = _wkey()
        assert "proc" not in k.as_filename().lower()

    def test_key_is_frozen(self) -> None:
        k = _wkey()
        with pytest.raises(AttributeError):
            k.blob_sha = "x" * 64  # type: ignore[misc]

    def test_auto_resample_default_false_absent_from_filename(self) -> None:
        assert ".rs" not in _wkey().as_filename()

    def test_auto_resample_true_adds_suffix(self) -> None:
        a = _wkey(auto_resample=False)
        b = _wkey(auto_resample=True)
        assert a.as_filename() != b.as_filename()
        assert ".rs" in b.as_filename()


class TestWaveformCacheRoundTrip:
    def test_miss_on_empty(self, tmp_path: Path) -> None:
        cache = WaveformCache(tmp_path / "wav")
        assert cache.get(_wkey()) is None

    def test_put_then_get(self, tmp_path: Path) -> None:
        cache = WaveformCache(tmp_path / "wav")
        waveform = np.arange(16_000 * 1, dtype=np.float32) / 16_000.0
        cache.put(_wkey(), waveform)
        loaded = cache.get(_wkey())
        assert loaded is not None
        np.testing.assert_array_equal(loaded, waveform)
        assert loaded.dtype == np.float32

    def test_put_creates_shard_dir(self, tmp_path: Path) -> None:
        cache = WaveformCache(tmp_path / "wav")
        key = _wkey(blob_sha="ef" + "0" * 62)
        cache.put(key, np.zeros((1,), dtype=np.float32))
        assert (tmp_path / "wav" / "ef").is_dir()

    def test_exists_flips_after_put(self, tmp_path: Path) -> None:
        cache = WaveformCache(tmp_path / "wav")
        key = _wkey()
        assert cache.exists(key) is False
        cache.put(key, np.zeros((1,), dtype=np.float32))
        assert cache.exists(key) is True

    def test_corrupt_file_treated_as_miss(self, tmp_path: Path) -> None:
        cache = WaveformCache(tmp_path / "wav")
        key = _wkey()
        cache.put(key, np.zeros((1,), dtype=np.float32))
        cache.path_for(key).write_bytes(b"not npz")
        assert cache.get(key) is None

    def test_clear_removes_tree(self, tmp_path: Path) -> None:
        cache = WaveformCache(tmp_path / "wav")
        cache.put(_wkey(), np.zeros((1,), dtype=np.float32))
        cache.clear()
        assert not (tmp_path / "wav").exists()


class TestWaveformAndFeatureCachesDistinct:
    """The two audio caches must not collide on-disk or in-memory."""

    def test_separate_roots_coexist(self, tmp_path: Path) -> None:
        features = AudioCache(tmp_path / "audio-cache")
        waveforms = WaveformCache(tmp_path / "audio-waveform-cache")
        features.put(
            AudioCacheKey(
                blob_sha="a" * 64,
                processor_sha="b" * 64,
                sample_rate=16_000,
                max_length_ms=30_000,
            ),
            np.zeros((1,), dtype=np.float32),
        )
        waveforms.put(_wkey(), np.zeros((1,), dtype=np.float32))
        # Neither cache's directory tree touches the other's.
        feat_files = set((tmp_path / "audio-cache").rglob("*.npz"))
        wave_files = set((tmp_path / "audio-waveform-cache").rglob("*.npz"))
        assert feat_files
        assert wave_files
        assert feat_files.isdisjoint(wave_files)
