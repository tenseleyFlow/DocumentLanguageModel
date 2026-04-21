"""Audio preprocessor tensor cache.

Mirrors `vl_cache.py`. Keyed on
`(blob_sha, processor_sha, sample_rate, max_length_seconds)` — a blob
bytes change, a processor / feature-extractor upgrade, a sample-rate
pin change, or a duration-cap change each invalidate the entry.
Orthogonal to the tokenized-section cache (which is keyed on
tokenizer sha, not audio processor sha).

Layout: `<audio-cache>/<blob_sha[:2]>/<blob_sha>.<proc_sha[:12]>.<sr>.<ms>.npz`.
Contents: a single numpy array stored under the key `input_features`.
Atomic write via `dlm.io.atomic.write_bytes` so a half-written file
never surfaces to a concurrent reader.

Processor identity (`processor_sha`) fingerprints the subset of
feature-extractor attributes that materially change output features:
`sampling_rate`, `feature_size`, `n_fft`, `hop_length`, `padding_value`,
and the class name. Full byte-level fingerprinting of the HF processor
isn't practical (they aren't JSON-clean); these fields match what the
Qwen2-Audio + Whisper-family feature extractors expose and what drift
between upstream revisions.
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

_FINGERPRINT_ATTR: Final[str] = "_dlm_audio_processor_sha256"


@dataclass(frozen=True)
class AudioCacheKey:
    """Composite key for one preprocessed audio tensor.

    `auto_resample` lands on the key (not just the preprocessor path)
    so a cached entry built without resampling isn't served to a
    caller that asked for auto-resample — the inputs to the processor
    differ when the source rate disagreed with the target.
    """

    blob_sha: str
    processor_sha: str
    sample_rate: int
    max_length_ms: int
    auto_resample: bool = False

    def as_filename(self) -> str:
        """Stable per-entry filename under the shard."""
        rs = ".rs" if self.auto_resample else ""
        return (
            f"{self.blob_sha}.{self.processor_sha[:12]}"
            f".{self.sample_rate}.{self.max_length_ms}{rs}.npz"
        )

    def shard(self) -> str:
        """First 2 hex chars of blob_sha — the directory shard."""
        return self.blob_sha[:2]


class AudioCache:
    """On-disk cache for preprocessed audio feature tensors.

    Lazy-initialized: constructing an `AudioCache` does not create the
    directory. The first `put` creates the root + shard on demand.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, key: AudioCacheKey) -> Path:
        return self._root / key.shard() / key.as_filename()

    def get(self, key: AudioCacheKey) -> np.ndarray | None:
        """Return the cached tensor, or `None` on miss."""
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            with np.load(path) as npz:
                arr: np.ndarray = npz["input_features"].copy()
                return arr
        except (OSError, KeyError, ValueError):
            # Corrupt entry — treat as miss; `dlm cache clear` sweeps.
            return None

    def put(self, key: AudioCacheKey, tensor: np.ndarray) -> Path:
        """Atomically write `tensor` under `key`; return the on-disk path."""
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.savez(buffer, input_features=tensor)
        write_bytes(path, buffer.getvalue())
        return path

    def exists(self, key: AudioCacheKey) -> bool:
        return self.path_for(key).exists()

    def clear(self) -> None:
        """Delete the entire cache tree. Test + opt-in user action only."""
        if self._root.exists():
            import shutil

            shutil.rmtree(self._root)


@dataclass(frozen=True)
class WaveformCacheKey:
    """Key for the training-hot-path waveform cache.

    Distinct from `AudioCacheKey` (feature-level, processor-dependent):
    waveforms are pre-processor, so the key has no `processor_sha` —
    any Qwen2-Audio / Whisper / Wav2Vec2 processor at the same pinned
    sample_rate + duration sees the same decoded + truncated waveform.
    The feature-extractor still runs per batch; the cache skips
    soundfile decode + mono-mixing + truncation on repeat epochs
    (which dominate per-batch CPU time on a small audio corpus).

    `auto_resample` lands on the key to separate native-rate entries
    from resampled ones — a 48 kHz file cached without resampling is
    not interchangeable with the same file resampled to 16 kHz.
    """

    blob_sha: str
    sample_rate: int
    max_length_ms: int
    auto_resample: bool = False

    def as_filename(self) -> str:
        rs = ".rs" if self.auto_resample else ""
        return f"{self.blob_sha}.{self.sample_rate}.{self.max_length_ms}{rs}.wav.npz"

    def shard(self) -> str:
        return self.blob_sha[:2]


class WaveformCache:
    """On-disk cache for decoded + mono-mixed + truncated waveforms.

    Parallel to `AudioCache` but keyed without `processor_sha` — the
    cached value is the pre-processor waveform (1-D float32 mono).
    Training's per-batch audio work is dominated by this decode step
    on a small corpus; caching turns a multi-epoch training run into
    a read-once + extract-each-epoch pattern.

    Stored as npz under key `waveform` so the on-disk layout stays
    distinct from `AudioCache`'s `input_features`.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, key: WaveformCacheKey) -> Path:
        return self._root / key.shard() / key.as_filename()

    def get(self, key: WaveformCacheKey) -> np.ndarray | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            with np.load(path) as npz:
                arr: np.ndarray = npz["waveform"].copy()
                return arr
        except (OSError, KeyError, ValueError):
            return None

    def put(self, key: WaveformCacheKey, waveform: np.ndarray) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.savez(buffer, waveform=waveform)
        write_bytes(path, buffer.getvalue())
        return path

    def exists(self, key: WaveformCacheKey) -> bool:
        return self.path_for(key).exists()

    def clear(self) -> None:
        if self._root.exists():
            import shutil

            shutil.rmtree(self._root)


def processor_sha256(processor: Any) -> str:
    """Canonical sha256 of the identity-bearing subset of an audio processor.

    The feature extractor (exposed as `processor.feature_extractor` on
    HF's `AutoProcessor` wrappers) carries the pre-processing params.
    Fingerprint a stable subset so an upstream bump that rewrites log-mel
    windowing invalidates every cached entry.

    Pinned on the processor instance via a private attribute for O(1)
    repeat calls within a run.
    """
    pinned: str | None = getattr(processor, _FINGERPRINT_ATTR, None)
    if pinned is not None:
        return pinned

    feature_extractor = getattr(processor, "feature_extractor", processor)
    state: dict[str, object] = {
        "class": processor.__class__.__name__,
        "fe_class": feature_extractor.__class__.__name__,
        "sampling_rate": _readable(getattr(feature_extractor, "sampling_rate", None)),
        "feature_size": _readable(getattr(feature_extractor, "feature_size", None)),
        "n_fft": _readable(getattr(feature_extractor, "n_fft", None)),
        "hop_length": _readable(getattr(feature_extractor, "hop_length", None)),
        "chunk_length": _readable(getattr(feature_extractor, "chunk_length", None)),
        "padding_value": _readable(getattr(feature_extractor, "padding_value", None)),
        "return_attention_mask": bool(getattr(feature_extractor, "return_attention_mask", False)),
    }
    canonical = json.dumps(state, sort_keys=True, default=str)
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with contextlib.suppress(AttributeError, TypeError):
        object.__setattr__(processor, _FINGERPRINT_ATTR, sha)
    return sha


def _readable(value: object) -> object:
    """Coerce a value into a JSON-serializable form (mirror of vl_cache)."""
    if value is None:
        return None
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_readable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _readable(v) for k, v in sorted(value.items())}
    return str(value)
