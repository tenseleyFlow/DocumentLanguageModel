"""VL preprocessor tensor cache.

Keyed on `(blob_sha, processor_sha, target_size)` — a blob-bytes
change, a processor upgrade, or a resize-policy bump each invalidate
the entry. Orthogonal to the tokenized-section cache: different
inputs, different consumers, different keys.

Layout: `<vl-cache>/<blob_sha[:2]>/<blob_sha>.<proc_sha[:12]>.<h>x<w>.npz`.
Contents: single numpy array stored under the key `pixel_values`.
Atomic write via `dlm.io.atomic.write_bytes` so a half-written file
never surfaces to a concurrent reader.

Processor identity (`processor_sha`) is derived from the subset of
attributes that materially change pixel output: `image_size`,
`image_mean`, `image_std`, and the class name. That's enough to
invalidate when a user upgrades HF transformers + the processor bumps
its normalization constants; full byte-level fingerprinting of the
processor isn't practical (processors aren't as JSON-clean as fast
tokenizers are).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from dlm.io.atomic import write_bytes

_FINGERPRINT_ATTR: Final[str] = "_dlm_processor_sha256"


@dataclass(frozen=True)
class VlCacheKey:
    """Composite key for one preprocessed image tensor."""

    blob_sha: str
    processor_sha: str
    target_height: int
    target_width: int

    def as_filename(self) -> str:
        """Stable per-entry filename under the shard."""
        return (
            f"{self.blob_sha}.{self.processor_sha[:12]}"
            f".{self.target_height}x{self.target_width}.npz"
        )

    def shard(self) -> str:
        """First 2 hex chars of blob_sha — the directory shard."""
        return self.blob_sha[:2]


class VlCache:
    """On-disk cache for preprocessed image tensors.

    Lazy-initialized: constructing a `VlCache` does not create the
    directory. The first `put` creates the root + shard on demand.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, key: VlCacheKey) -> Path:
        return self._root / key.shard() / key.as_filename()

    def get(self, key: VlCacheKey) -> np.ndarray | None:
        """Return the cached tensor, or `None` on miss."""
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            with np.load(path) as npz:
                arr: np.ndarray = npz["pixel_values"].copy()
                return arr
        except (OSError, KeyError, ValueError):
            # Corrupt cache entry — treat as miss so the trainer can
            # re-tokenize. The stale file stays on disk for `dlm cache
            # clear` to sweep rather than racing a delete here.
            return None

    def put(self, key: VlCacheKey, tensor: np.ndarray) -> Path:
        """Atomically write `tensor` under `key`; return the on-disk path."""
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.savez(buffer, pixel_values=tensor)
        write_bytes(path, buffer.getvalue())
        return path

    def exists(self, key: VlCacheKey) -> bool:
        return self.path_for(key).exists()

    def clear(self) -> None:
        """Delete the entire cache tree. Test + opt-in user action only."""
        if self._root.exists():
            import shutil

            shutil.rmtree(self._root)


def processor_sha256(processor: Any) -> str:
    """Canonical sha256 of the identity-bearing subset of a processor.

    HF `AutoProcessor` instances aren't JSON-serializable, so we
    fingerprint the attributes that actually drive pixel output:
    `image_size` (or `size` mapping), `image_mean`, `image_std`, and
    the class name. A future bump in any of these invalidates the
    cache exactly like a tokenizer-fingerprint change does for text.

    Pinned on the processor instance via a private attribute for O(1)
    repeat calls within a run.
    """
    pinned: str | None = getattr(processor, _FINGERPRINT_ATTR, None)
    if pinned is not None:
        return pinned

    image_processor = getattr(processor, "image_processor", processor)
    state: dict[str, object] = {
        "class": processor.__class__.__name__,
        "image_size": _readable(getattr(image_processor, "image_size", None)),
        "size": _readable(getattr(image_processor, "size", None)),
        "image_mean": _readable(getattr(image_processor, "image_mean", None)),
        "image_std": _readable(getattr(image_processor, "image_std", None)),
        "do_normalize": bool(getattr(image_processor, "do_normalize", True)),
        "do_rescale": bool(getattr(image_processor, "do_rescale", True)),
        "rescale_factor": _readable(getattr(image_processor, "rescale_factor", None)),
        "resample": _readable(getattr(image_processor, "resample", None)),
    }
    canonical = json.dumps(state, sort_keys=True, default=str)
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with contextlib.suppress(AttributeError, TypeError):
        object.__setattr__(processor, _FINGERPRINT_ATTR, sha)
    return sha


def _readable(value: object) -> object:
    """Coerce a value into a JSON-serializable form.

    HF processors use mixed types — ints, floats, lists, dicts, enum
    members (`PILImageResampling`). Stringify exotic types so the
    fingerprint stays stable across HF version bumps that rewrap an
    int as an enum member.
    """
    if value is None:
        return None
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_readable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _readable(v) for k, v in sorted(value.items())}
    return str(value)
