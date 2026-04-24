"""Audio waveform resampling.

Opt-in helper driven by `training.audio.auto_resample=True`. Off the
happy path the preprocessor + collator still refuse on SR mismatch
(v11 contract); `auto_resample=True` flips those hard errors to a
resample-on-decode call through `resample`.

Two backends, preferred in order:

1. **soxr** — libsoxr bindings (`soxr.resample`). High-quality
   polyphase resampler, written in C, ~10× faster than scipy on
   typical audio. Optional because it requires the libsoxr native
   library (ships as a wheel on most platforms).
2. **scipy.signal.resample_poly** — pure-Python fallback using
   scipy's polyphase implementation. Always available when scipy is
   installed. Slightly lower quality than soxr but still high.

If neither is importable, `resample` raises `AudioResampleUnavailable`
with an actionable install hint. Callers should surface this at
plan-resolve time rather than letting a training loop crash mid-run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from dlm.data.errors import DataError

_Backend = Callable[..., np.ndarray]


class AudioResampleUnavailable(DataError):  # noqa: N818 — mirrors DataError sibling naming
    """Neither soxr nor scipy is importable for `training.audio.auto_resample=True`.

    Suggests the two install paths: ``pip install soxr`` (preferred,
    libsoxr native lib) or ``pip install scipy`` (pure-Python
    fallback). Surfaces once at backend-probe time, never mid-batch.
    """


def resample(waveform: np.ndarray, *, src_sr: int, dst_sr: int) -> np.ndarray:
    """Return `waveform` resampled from `src_sr` → `dst_sr`.

    Input and output are 1-D float32 mono arrays. If `src_sr == dst_sr`
    returns the input unchanged (no copy). On rate change, routes
    through soxr if importable, else scipy.signal.resample_poly. No
    silent fallback beyond that — if neither backend is available the
    call raises rather than returning the un-resampled waveform (that
    would train on the wrong sample rate, a silent correctness bug).
    """
    if src_sr == dst_sr:
        return waveform
    if src_sr <= 0 or dst_sr <= 0:
        raise ValueError(
            f"resample: sample rates must be positive, got src_sr={src_sr} dst_sr={dst_sr}"
        )

    backend = _pick_backend()
    return backend(waveform, src_sr=src_sr, dst_sr=dst_sr)


def _pick_backend() -> _Backend:
    """Resolve the first importable resampler. Raises when none found.

    Probes each backend's actual import path rather than returning a
    wrapper that fails later — surfacing the missing-dep error at
    backend-pick time keeps the failure near the user's config.
    """
    try:
        import soxr  # noqa: F401
    except ImportError:
        pass
    else:
        return _soxr_resample

    try:
        import scipy.signal  # noqa: F401
    except ImportError:
        pass
    else:
        return _scipy_resample

    raise AudioResampleUnavailable(
        "training.audio.auto_resample=True requires either soxr or scipy; "
        "install one of: `pip install soxr` (recommended) or "
        "`pip install scipy`. Until then re-encode the audio files to "
        "the base's pinned rate manually with `ffmpeg -i <in> -ar <sr> <out>`."
    )


def _soxr_resample(waveform: np.ndarray, *, src_sr: int, dst_sr: int) -> np.ndarray:
    """soxr backend. Highest quality + speed, requires libsoxr wheel."""
    import soxr

    out: Any = soxr.resample(waveform, src_sr, dst_sr, quality="HQ")
    return np.ascontiguousarray(out, dtype=np.float32)


def _scipy_resample(waveform: np.ndarray, *, src_sr: int, dst_sr: int) -> np.ndarray:
    """scipy.signal.resample_poly fallback.

    Reduces (src_sr, dst_sr) to their coprime pair so the polyphase
    filter uses the minimal up/down factors. scipy handles any
    integer ratio; non-integer ratios reduce the same way.
    """
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(src_sr, dst_sr)
    up = dst_sr // divisor
    down = src_sr // divisor
    out: Any = resample_poly(waveform, up=up, down=down)
    return np.ascontiguousarray(out, dtype=np.float32)
