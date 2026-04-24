"""VL cache — key stability, atomic I/O, processor fingerprint.

Covers:

- `VlCacheKey.as_filename` / `shard` are deterministic.
- Different target_size / processor_sha produce different filenames.
- Round-trip: put → get returns byte-identical array.
- Miss on empty store, miss on corrupt file.
- `processor_sha256` is stable across repeat calls + pinned on instance.
- Different preprocessor constants drift the sha.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dlm.data.vl_cache import VlCache, VlCacheKey, processor_sha256


def _key(**overrides: object) -> VlCacheKey:
    defaults = {
        "blob_sha": "a" * 64,
        "processor_sha": "b" * 64,
        "target_height": 224,
        "target_width": 224,
    }
    defaults.update(overrides)
    return VlCacheKey(**defaults)  # type: ignore[arg-type]


class TestVlCacheKey:
    def test_filename_shape(self) -> None:
        key = _key()
        assert key.as_filename() == f"{'a' * 64}.{'b' * 12}.224x224.npz"

    def test_shard_is_two_prefix(self) -> None:
        assert _key(blob_sha="cd" + "0" * 62).shard() == "cd"

    def test_different_size_different_filename(self) -> None:
        a = _key(target_height=224)
        b = _key(target_height=336)
        assert a.as_filename() != b.as_filename()

    def test_different_processor_different_filename(self) -> None:
        a = _key(processor_sha="1" * 64)
        b = _key(processor_sha="2" * 64)
        assert a.as_filename() != b.as_filename()

    def test_key_is_frozen(self) -> None:
        key = _key()
        with pytest.raises(AttributeError):
            key.blob_sha = "x" * 64  # type: ignore[misc]


class TestVlCacheRoundTrip:
    def test_miss_on_empty(self, tmp_path: Path) -> None:
        cache = VlCache(tmp_path / "vl")
        assert cache.get(_key()) is None

    def test_put_then_get(self, tmp_path: Path) -> None:
        cache = VlCache(tmp_path / "vl")
        tensor = np.arange(3 * 4 * 5, dtype=np.float32).reshape(1, 3, 4, 5)
        cache.put(_key(), tensor)
        loaded = cache.get(_key())
        assert loaded is not None
        np.testing.assert_array_equal(loaded, tensor)
        assert loaded.dtype == np.float32

    def test_put_creates_shard_dir(self, tmp_path: Path) -> None:
        cache = VlCache(tmp_path / "vl")
        key = _key(blob_sha="ef" + "0" * 62)
        cache.put(key, np.zeros((1,), dtype=np.float32))
        assert (tmp_path / "vl" / "ef").is_dir()

    def test_exists_flips_after_put(self, tmp_path: Path) -> None:
        cache = VlCache(tmp_path / "vl")
        key = _key()
        assert cache.exists(key) is False
        cache.put(key, np.zeros((1,), dtype=np.float32))
        assert cache.exists(key) is True

    def test_corrupt_file_treated_as_miss(self, tmp_path: Path) -> None:
        cache = VlCache(tmp_path / "vl")
        key = _key()
        cache.put(key, np.zeros((1,), dtype=np.float32))
        # Corrupt on disk.
        cache.path_for(key).write_bytes(b"not a real npz")
        assert cache.get(key) is None

    def test_clear_removes_tree(self, tmp_path: Path) -> None:
        cache = VlCache(tmp_path / "vl")
        cache.put(_key(), np.zeros((1,), dtype=np.float32))
        cache.clear()
        assert not (tmp_path / "vl").exists()


class TestProcessorSha256:
    def _make_processor(self, **attrs: object) -> SimpleNamespace:
        defaults: dict[str, object] = {
            "image_size": (224, 224),
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "do_normalize": True,
            "do_rescale": True,
            "rescale_factor": 1 / 255,
            "resample": 2,
        }
        defaults.update(attrs)
        return SimpleNamespace(**defaults)

    def test_stable_across_calls(self) -> None:
        proc = self._make_processor()
        assert processor_sha256(proc) == processor_sha256(proc)

    def test_pinned_on_instance(self) -> None:
        proc = self._make_processor()
        first = processor_sha256(proc)
        # Mutate a field that would drift the sha if recomputed — the
        # pinned cache returns the original so repeat calls stay O(1).
        proc.image_mean = [0.1, 0.1, 0.1]
        assert processor_sha256(proc) == first

    def test_different_size_different_sha(self) -> None:
        a = self._make_processor(image_size=(224, 224))
        b = self._make_processor(image_size=(336, 336))
        assert processor_sha256(a) != processor_sha256(b)

    def test_different_mean_different_sha(self) -> None:
        a = self._make_processor(image_mean=[0.5, 0.5, 0.5])
        b = self._make_processor(image_mean=[0.1, 0.2, 0.3])
        assert processor_sha256(a) != processor_sha256(b)

    def test_different_class_different_sha(self) -> None:
        class ProcA:
            image_size = (224, 224)
            image_mean = [0.5] * 3
            image_std = [0.5] * 3

        class ProcB:
            image_size = (224, 224)
            image_mean = [0.5] * 3
            image_std = [0.5] * 3

        assert processor_sha256(ProcA()) != processor_sha256(ProcB())

    def test_nested_dict_and_tuple_fields_are_readable(self) -> None:
        proc = SimpleNamespace(
            image_processor=SimpleNamespace(
                size={"shortest_edge": 224, "crop": (224, 224)},
                image_mean=(0.5, 0.5, 0.5),
                image_std=[0.2, 0.2, 0.2],
                do_normalize=True,
                do_rescale=True,
                rescale_factor=1 / 255,
                resample="bicubic",
            )
        )
        sha = processor_sha256(proc)
        assert len(sha) == 64

    def test_exotic_resample_value_stringifies_stably(self) -> None:
        proc = SimpleNamespace(
            image_processor=SimpleNamespace(
                size={"shortest_edge": 224},
                image_mean=[0.5] * 3,
                image_std=[0.5] * 3,
                do_normalize=True,
                do_rescale=True,
                rescale_factor=1 / 255,
                resample=object(),
            )
        )
        sha = processor_sha256(proc)
        assert len(sha) == 64
