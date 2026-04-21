"""Audio preprocessing: blob bytes → feature tensor via HF AutoProcessor.

Thin wrapper that loads an audio file via `soundfile`, enforces the
spec's pinned `sample_rate`, truncates to `max_length_seconds`, and
runs the HF processor. On-disk caching is keyed on
`(blob_sha, processor_sha, sample_rate, max_length_seconds)`.

Callers own the processor lifecycle — `AutoProcessor.from_pretrained`
is expensive, so loading it once at trainer startup and reusing across
sections is the expected pattern. The cache does the heavy lifting for
repeat runs on the same corpus.

**v1 policy: refuse on sample-rate mismatch.** Resampling lands as a
35.2 follow-up once we agree on the resampler (soxr is the leading
candidate; pytorch's sample-rate conversion is available but inflates
the install surface on CPU-only hosts). Refusing early produces an
actionable error instead of a silently-corrupt training run.

Heavy imports (`soundfile`, `numpy`) happen inside the functions that
need them; the module is cheap to import for CLI subcommands that
don't touch audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from dlm.data.audio_cache import AudioCache, AudioCacheKey, processor_sha256
from dlm.data.errors import DataError


class AudioSampleRateMismatch(DataError):  # noqa: N818 — `*Mismatch` mirrors other DataError subclasses
    """Audio file sample rate doesn't match the base's pinned value.

    Sprint 35.2 v1 refuses rather than resampling silently. The error
    message echoes both rates so the user can re-encode with `ffmpeg
    -ar <target>` or pick a base pinned to the clip's native rate.
    """


@dataclass(frozen=True)
class PreprocessedAudio:
    """Result of running a processor over a single audio clip.

    `input_features` is the processor's mel/log-mel tensor shaped
    `(num_mel_bins, num_frames)` for most audio-LM bases. `cache_hit`
    records whether the value came from disk so callers can surface
    hit rates (parallel to the VL preprocessor).
    """

    input_features: np.ndarray
    cache_hit: bool


_CACHE_KEY_FACTORY: Final = AudioCacheKey


def preprocess_audio(
    *,
    blob_path: Path,
    blob_sha: str,
    processor: Any,
    sample_rate: int,
    max_length_seconds: float,
    cache: AudioCache | None = None,
) -> PreprocessedAudio:
    """Preprocess one audio blob into a feature tensor.

    `processor` is a pre-loaded HF processor. `sample_rate` and
    `max_length_seconds` come from the base's `AudioPreprocessorPlan`
    — they pin both the refusal gate *and* the cache key.

    On cache hit, returns the cached array without touching the
    processor. On miss, reads the file via `soundfile`, refuses if
    the rates disagree, truncates to the max duration, runs the
    processor, and writes the result back through the cache.
    `cache=None` bypasses caching entirely (tests, ad-hoc prompts).

    Raises `AudioSampleRateMismatch` when the file's native sample
    rate disagrees with the spec's pinned rate.
    """
    proc_sha = processor_sha256(processor)
    key = _CACHE_KEY_FACTORY(
        blob_sha=blob_sha,
        processor_sha=proc_sha,
        sample_rate=sample_rate,
        max_length_ms=int(round(max_length_seconds * 1000)),
    )

    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return PreprocessedAudio(input_features=hit, cache_hit=True)

    tensor = _run_processor(
        processor,
        blob_path,
        target_sample_rate=sample_rate,
        max_length_seconds=max_length_seconds,
    )

    if cache is not None:
        cache.put(key, tensor)

    return PreprocessedAudio(input_features=tensor, cache_hit=False)


def _run_processor(
    processor: Any,
    blob_path: Path,
    *,
    target_sample_rate: int,
    max_length_seconds: float,
) -> np.ndarray:
    """Drive the HF processor over one audio clip, return features.

    Loads the waveform via `soundfile` as float32 mono (average
    channels if stereo), refuses on sample-rate mismatch, truncates
    to `max_length_seconds * target_sample_rate` samples, then passes
    through `processor(audios=..., sampling_rate=..., return_tensors="np")`.
    """
    import soundfile as sf  # type: ignore[import-untyped]

    data, native_sr = sf.read(str(blob_path), dtype="float32", always_2d=False)
    if native_sr != target_sample_rate:
        raise AudioSampleRateMismatch(
            f"audio {blob_path.name!r}: native sample_rate={native_sr} Hz "
            f"does not match pinned {target_sample_rate} Hz "
            "(resampling lands in a 35.2 follow-up; re-encode with "
            f"`ffmpeg -i <in> -ar {target_sample_rate} <out>` for now)"
        )

    # Mono-ize: average across channels if stereo+.
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32, copy=False)
    data = np.ascontiguousarray(data, dtype=np.float32)

    max_samples = int(round(max_length_seconds * target_sample_rate))
    if data.shape[0] > max_samples:
        data = data[:max_samples]

    outputs = processor(
        audios=data,
        sampling_rate=target_sample_rate,
        return_tensors="np",
    )
    input_features = outputs["input_features"]
    if not isinstance(input_features, np.ndarray):
        input_features = np.asarray(input_features, dtype=np.float32)
    result: np.ndarray = input_features.astype(np.float32, copy=False)
    return result
