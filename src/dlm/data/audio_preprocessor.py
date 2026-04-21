"""Audio preprocessing: blob bytes → feature tensor via HF AutoProcessor.

Thin wrapper that loads an audio file via `soundfile`, reconciles with
the spec's pinned `sample_rate` (refuse by default; resample on opt-in),
truncates to `max_length_seconds`, and runs the HF processor. On-disk
caching is keyed on `(blob_sha, processor_sha, sample_rate,
max_length_seconds, auto_resample)` — the flag lands on the key so
cached native-rate entries don't serve resample-opted-in callers.

Callers own the processor lifecycle — `AutoProcessor.from_pretrained`
is expensive, so loading it once at trainer startup and reusing across
sections is the expected pattern. The cache does the heavy lifting for
repeat runs on the same corpus.

**Sample-rate mismatch policy.** Default `auto_resample=False` preserves
the original contract: raise `AudioSampleRateMismatch` on rate disagree.
`auto_resample=True` (flipped via `training.audio.auto_resample`)
routes through `dlm.data.audio_resample` which raises
`AudioResampleUnavailable` if neither soxr nor scipy is importable.
Both failure modes surface actionable errors rather than silently
training on the wrong rate.

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
    auto_resample: bool = False,
) -> PreprocessedAudio:
    """Preprocess one audio blob into a feature tensor.

    `processor` is a pre-loaded HF processor. `sample_rate` and
    `max_length_seconds` come from the base's `AudioPreprocessorPlan`
    — they pin both the reconciliation gate *and* the cache key.

    On cache hit, returns the cached array without touching the
    processor. On miss, reads the file via `soundfile`, reconciles
    against the target rate (refuse when `auto_resample=False`,
    resample when `auto_resample=True`), truncates to the max
    duration, runs the processor, and writes the result back through
    the cache. `cache=None` bypasses caching entirely (tests, ad-hoc
    prompts).

    Raises `AudioSampleRateMismatch` when rates disagree and
    `auto_resample=False`; raises `AudioResampleUnavailable` when
    `auto_resample=True` but neither soxr nor scipy is importable.
    """
    proc_sha = processor_sha256(processor)
    key = _CACHE_KEY_FACTORY(
        blob_sha=blob_sha,
        processor_sha=proc_sha,
        sample_rate=sample_rate,
        max_length_ms=int(round(max_length_seconds * 1000)),
        auto_resample=auto_resample,
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
        auto_resample=auto_resample,
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
    auto_resample: bool = False,
) -> np.ndarray:
    """Drive the HF processor over one audio clip, return features.

    Loads the waveform via `soundfile` as float32 mono (average
    channels if stereo), reconciles against `target_sample_rate`
    (refuse when `auto_resample=False`, resample when `True`),
    truncates to `max_length_seconds * target_sample_rate` samples,
    then passes through
    `processor(audios=..., sampling_rate=..., return_tensors="np")`.
    """
    import soundfile as sf  # type: ignore[import-untyped]

    data, native_sr = sf.read(str(blob_path), dtype="float32", always_2d=False)

    # Mono-ize first: resampling a stereo waveform then mixing can
    # smear channel-specific transients; mixing before resampling
    # keeps the resampler's anti-alias filter well-behaved.
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32, copy=False)
    data = np.ascontiguousarray(data, dtype=np.float32)

    if native_sr != target_sample_rate:
        if not auto_resample:
            raise AudioSampleRateMismatch(
                f"audio {blob_path.name!r}: native sample_rate={native_sr} Hz "
                f"does not match pinned {target_sample_rate} Hz. "
                f"Set `training.audio.auto_resample: true` to resample on "
                f"the fly, or re-encode manually with "
                f"`ffmpeg -i <in> -ar {target_sample_rate} <out>`."
            )
        from dlm.data.audio_resample import resample

        data = resample(data, src_sr=native_sr, dst_sr=target_sample_rate)

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
