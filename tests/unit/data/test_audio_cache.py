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

from dlm.data.audio_cache import AudioCache, AudioCacheKey, processor_sha256


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
