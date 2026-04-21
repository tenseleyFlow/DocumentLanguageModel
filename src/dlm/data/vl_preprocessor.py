"""Vision preprocessing: blob bytes → tensor via HF AutoProcessor.

Thin wrapper that runs a pre-loaded HF processor over a PIL image
loaded from the content-addressed blob store, with on-disk caching
keyed on `(blob_sha, processor_sha, target_size)`.

Callers own the processor lifecycle — `AutoProcessor.from_pretrained`
is expensive, so loading it once at trainer startup and reusing
across sections is the expected pattern. The cache does the heavy
lifting for repeat runs on the same corpus.

Heavy imports (`PIL`, `numpy`) happen inside the functions that
need them; the module is cheap to import for CLI subcommands that
don't touch images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from dlm.data.vl_cache import VlCache, VlCacheKey, processor_sha256


@dataclass(frozen=True)
class PreprocessedImage:
    """Result of running a processor over a single image.

    `pixel_values` is the processor's pixel tensor shaped
    `(num_patches, channels, height, width)` for most VL bases; some
    (Qwen2-VL) emit variable patch counts per image. `cache_hit`
    records whether the value came from disk so callers can surface
    hit rates.
    """

    pixel_values: np.ndarray
    cache_hit: bool


_CACHE_KEY_FACTORY: Final = VlCacheKey


def preprocess_image(
    *,
    blob_path: Path,
    blob_sha: str,
    processor: Any,
    target_size: tuple[int, int],
    cache: VlCache | None = None,
) -> PreprocessedImage:
    """Preprocess a single image blob into a pixel-values tensor.

    `processor` is a pre-loaded HF processor (`AutoProcessor.from_pretrained`).
    `target_size` is the pinned `(height, width)` from the base's
    `VlPreprocessorPlan` — part of the cache key.

    On cache hit, returns the cached array without touching the
    processor. On miss, runs the processor and writes the result
    back through the cache. `cache=None` bypasses caching entirely
    (tests, ad-hoc prompts).
    """
    proc_sha = processor_sha256(processor)
    key = _CACHE_KEY_FACTORY(
        blob_sha=blob_sha,
        processor_sha=proc_sha,
        target_height=target_size[0],
        target_width=target_size[1],
    )

    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return PreprocessedImage(pixel_values=hit, cache_hit=True)

    tensor = _run_processor(processor, blob_path)

    if cache is not None:
        cache.put(key, tensor)

    return PreprocessedImage(pixel_values=tensor, cache_hit=False)


def _run_processor(processor: Any, blob_path: Path) -> np.ndarray:
    """Drive the HF processor over one image, return `pixel_values` array.

    Loads the image lazily via PIL, closes it immediately after the
    processor call so file handles don't pile up on large corpora.
    Returns a float32 numpy array — HF processors default to torch
    tensors when available, so the return path coerces explicitly.
    """
    from PIL import Image

    with Image.open(blob_path) as pil_image:
        pil_image.load()
        rgb = pil_image.convert("RGB")

    outputs = processor(images=rgb, return_tensors="np")
    pixel_values = outputs["pixel_values"]
    if not isinstance(pixel_values, np.ndarray):
        # Defensive: processor honored return_tensors but wrapped as
        # a torch tensor anyway (some versions of some processors).
        pixel_values = np.asarray(pixel_values, dtype=np.float32)
    result: np.ndarray = pixel_values.astype(np.float32, copy=False)
    return result
