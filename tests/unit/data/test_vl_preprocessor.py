"""VL preprocessor — cache hit/miss + processor dispatch.

Uses a stub processor so tests don't require the full transformers
import path. The wiring into HF `AutoProcessor` is exercised in the
slow VL integration test (T12).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dlm.data.vl_cache import VlCache
from dlm.data.vl_preprocessor import preprocess_image


class _StubProcessor:
    """Deterministic fake processor.

    `image_size` / `image_mean` / `image_std` drive the fingerprint;
    `calls` counts how many times the processor actually ran so tests
    can assert cache hits bypass it.
    """

    image_size = (224, 224)
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_normalize = True
    do_rescale = True
    rescale_factor = 1 / 255
    resample = 2

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, images: Any, return_tensors: str) -> dict[str, np.ndarray]:
        self.calls += 1
        _ = images
        _ = return_tensors
        # Emit a deterministic (1, 3, 2, 2) float32 array so cache
        # round-trip checks have something to compare.
        return {
            "pixel_values": np.full(
                (1, 3, 2, 2),
                float(self.calls),
                dtype=np.float32,
            ),
        }


@pytest.fixture
def tiny_png(tmp_path: Path) -> Path:
    """1×1 PNG on disk. Minimal PIL-openable payload."""
    from PIL import Image

    path = tmp_path / "pixel.png"
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(path, format="PNG")
    return path


class TestPreprocessImageNoCache:
    def test_runs_processor(self, tiny_png: Path) -> None:
        proc = _StubProcessor()
        result = preprocess_image(
            blob_path=tiny_png,
            blob_sha="a" * 64,
            processor=proc,
            target_size=(224, 224),
            cache=None,
        )
        assert proc.calls == 1
        assert result.cache_hit is False
        assert result.pixel_values.shape == (1, 3, 2, 2)
        assert result.pixel_values.dtype == np.float32


class TestPreprocessImageWithCache:
    def test_first_call_misses_then_writes(
        self,
        tiny_png: Path,
        tmp_path: Path,
    ) -> None:
        proc = _StubProcessor()
        cache = VlCache(tmp_path / "vl")
        result = preprocess_image(
            blob_path=tiny_png,
            blob_sha="a" * 64,
            processor=proc,
            target_size=(224, 224),
            cache=cache,
        )
        assert result.cache_hit is False
        assert proc.calls == 1
        assert any(cache.root.rglob("*.npz"))

    def test_second_call_hits_cache(
        self,
        tiny_png: Path,
        tmp_path: Path,
    ) -> None:
        proc = _StubProcessor()
        cache = VlCache(tmp_path / "vl")
        first = preprocess_image(
            blob_path=tiny_png,
            blob_sha="a" * 64,
            processor=proc,
            target_size=(224, 224),
            cache=cache,
        )
        second = preprocess_image(
            blob_path=tiny_png,
            blob_sha="a" * 64,
            processor=proc,
            target_size=(224, 224),
            cache=cache,
        )
        assert second.cache_hit is True
        # Processor ran exactly once; the second call came from disk.
        assert proc.calls == 1
        # Cache-hit array is byte-identical.
        np.testing.assert_array_equal(first.pixel_values, second.pixel_values)

    def test_different_target_size_misses(
        self,
        tiny_png: Path,
        tmp_path: Path,
    ) -> None:
        proc = _StubProcessor()
        cache = VlCache(tmp_path / "vl")
        preprocess_image(
            blob_path=tiny_png,
            blob_sha="a" * 64,
            processor=proc,
            target_size=(224, 224),
            cache=cache,
        )
        preprocess_image(
            blob_path=tiny_png,
            blob_sha="a" * 64,
            processor=proc,
            target_size=(336, 336),
            cache=cache,
        )
        assert proc.calls == 2

    def test_different_blob_sha_misses(
        self,
        tiny_png: Path,
        tmp_path: Path,
    ) -> None:
        proc = _StubProcessor()
        cache = VlCache(tmp_path / "vl")
        preprocess_image(
            blob_path=tiny_png,
            blob_sha="a" * 64,
            processor=proc,
            target_size=(224, 224),
            cache=cache,
        )
        preprocess_image(
            blob_path=tiny_png,
            blob_sha="b" * 64,
            processor=proc,
            target_size=(224, 224),
            cache=cache,
        )
        assert proc.calls == 2


class TestPreprocessImageReturnsNumpy:
    def test_coerces_non_ndarray(self, tiny_png: Path) -> None:
        class WrappedProc(_StubProcessor):
            def __call__(self, *, images: Any, return_tensors: str) -> dict[str, list]:  # type: ignore[override]
                self.calls += 1
                return {"pixel_values": [[[[1.0, 2.0], [3.0, 4.0]]]]}

        proc = WrappedProc()
        result = preprocess_image(
            blob_path=tiny_png,
            blob_sha="c" * 64,
            processor=proc,
            target_size=(224, 224),
            cache=None,
        )
        assert isinstance(result.pixel_values, np.ndarray)
        assert result.pixel_values.dtype == np.float32
