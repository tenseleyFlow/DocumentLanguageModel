"""Custom data collator for audio-language training.

TRL 1.2 ships `DataCollatorForVisionLanguageModeling` for VL bases but
does **not** ship an audio equivalent. This module fills the gap: it
takes the path-based audio rows emitted by `sections_to_rows` and
turns a list-of-rows batch into the `input_ids / attention_mask /
labels / input_features / feature_attention_mask` dict that
`Qwen2AudioForConditionalGeneration` (and any future audio-LM class
with a similar processor contract) expects.

Design choices:

- Rows carry `audio_path` + `audio_blob_sha` (not decoded waveforms).
  This keeps the HF `Dataset` rows small and lets the collator decide
  whether to decode per-batch or hit the cache.
- The optional `WaveformCache` memoizes the
  `soundfile decode → mono-mix → truncate` pipeline on disk, keyed on
  `(blob_sha, sample_rate, max_length_ms)`. The HF
  processor's feature extractor still runs every batch — caching its
  output would require re-implementing Qwen2-Audio's text-expansion
  logic (the processor derives per-audio placeholder counts from
  `feature_attention_mask`). The waveform cache covers the step that
  actually dominates per-batch CPU time on a small corpus (decoding
  a 30 s .wav is a few hundred ms; re-running on every epoch adds
  up).
- Labels = `input_ids` with pad positions masked to `-100`. This is
  full-sequence training (the model predicts every non-pad token
  including the audio placeholder expansion). Instruction-tuning
  variants that mask the audio + prompt to train only the response
  land as a follow-up; the simpler shape is enough to get signal from
  a small audio corpus and matches several published recipes.
- The HF processor owns audio-token-placeholder expansion — we pass
  `text` verbatim and the processor replaces `<|AUDIO|>` with the
  correct number of placeholder tokens derived from the audio frame
  count. Our pinned `max_length_seconds` keeps that count stable.

The collator is deliberately not a `@dataclass` because TRL's trainer
callbacks sometimes introspect collator attributes; keeping plain
`__init__` state avoids surprises.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from dlm.data.audio_cache import WaveformCache, WaveformCacheKey

_LOG = logging.getLogger(__name__)

_IGNORE_INDEX = -100  # HF convention — CrossEntropyLoss skips these positions


class AudioLmCollator:
    """Collator for path-based audio rows → HF model-ready batch dict.

    Parameters
    ----------
    processor:
        Loaded `AutoProcessor` (e.g. Qwen2AudioProcessor). Must expose
        `.tokenizer` with a pad token set.
    sample_rate:
        Target sample rate in Hz (from `AudioPreprocessorPlan`).
        Rows whose native rate disagrees raise by default (same gate
        as `preprocess_audio`). Pass `auto_resample=True` to resample
        on the fly via `dlm.data.audio_resample`.
    max_length_seconds:
        Per-clip duration cap in seconds (from
        `AudioPreprocessorPlan`). Longer waveforms are truncated.
    max_length:
        Optional token-length cap for the text side (post-expansion).
        `None` uses the processor's built-in limit.
    waveform_cache:
        Optional `WaveformCache` for memoizing decoded + mono-mixed
        + truncated waveforms across training epochs. `None` decodes
        fresh every batch (the pre-deferred behavior). Cache keys
        carry `auto_resample` so native-rate and resampled entries
        don't collide.
    auto_resample:
        Opt-in flag flipped by `training.audio.auto_resample=True`.
        When True, SR-mismatched files resample to `sample_rate`
        instead of raising. Requires soxr or scipy; absence surfaces
        as `AudioResampleUnavailable` at first mismatched decode.
    """

    def __init__(
        self,
        *,
        processor: Any,
        sample_rate: int,
        max_length_seconds: float,
        max_length: int | None = None,
        waveform_cache: WaveformCache | None = None,
        auto_resample: bool = False,
    ) -> None:
        self._processor = processor
        self._sample_rate = sample_rate
        self._max_length_seconds = max_length_seconds
        self._max_length = max_length
        self._waveform_cache = waveform_cache
        self._auto_resample = auto_resample
        self._max_length_ms = int(round(max_length_seconds * 1000))

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "AudioLmCollator: processor has no `.tokenizer` attribute — "
                "cannot resolve pad token id"
            )
        self._pad_token_id = tokenizer.pad_token_id
        if self._pad_token_id is None:
            raise ValueError(
                "AudioLmCollator: tokenizer has no pad_token_id — "
                "prepare_tokenizer must run before the collator is built"
            )

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Turn a list of dataset rows into a model-ready batch dict."""
        if not rows:
            raise ValueError("AudioLmCollator: received an empty batch")
        texts: list[str] = []
        waveforms: list[np.ndarray] = []
        for row in rows:
            if "audio_path" not in row or "text" not in row:
                raise ValueError(
                    "AudioLmCollator: row is missing required keys "
                    f"({set(row.keys())}); expected audio_path + text"
                )
            blob_sha = row.get("audio_blob_sha")
            waveforms.append(self._load_waveform(Path(row["audio_path"]), blob_sha=blob_sha))
            texts.append(row["text"])

        # One processor call over the whole batch: it handles padding
        # across both the token side (pad to max_length) and the audio-
        # feature side (pad to longest spectrogram). `return_tensors="pt"`
        # gives us torch tensors matching the model's expected dtypes.
        batch = self._processor(
            text=texts,
            audios=waveforms,
            sampling_rate=self._sample_rate,
            return_tensors="pt",
            padding=True,
            **({"max_length": self._max_length} if self._max_length else {}),
        )

        import torch as _torch

        input_ids: _torch.Tensor = batch["input_ids"]
        labels = input_ids.clone()
        labels[labels == self._pad_token_id] = _IGNORE_INDEX
        batch["labels"] = labels
        return dict(batch)

    def _load_waveform(self, path: Path, *, blob_sha: str | None = None) -> np.ndarray:
        """Decode one audio blob into a mono float32 waveform.

        When `waveform_cache` is configured and `blob_sha` is provided,
        hits the on-disk cache keyed on
        `(blob_sha, sample_rate, max_length_ms)`. Cache miss → decode
        via `soundfile`, mono-mix, truncate, populate cache.

        Refuses on sample-rate mismatch (same gate as
        `preprocess_audio`). Stereo-to-mono by channel averaging.
        Truncates to the configured duration.
        """
        # Cache lookup: only when both the cache is configured and the
        # row carries a blob sha (older row shapes may not have one;
        # skip the cache rather than break them).
        cache_key: WaveformCacheKey | None = None
        if self._waveform_cache is not None and blob_sha is not None:
            cache_key = WaveformCacheKey(
                blob_sha=blob_sha,
                sample_rate=self._sample_rate,
                max_length_ms=self._max_length_ms,
                auto_resample=self._auto_resample,
            )
            hit = self._waveform_cache.get(cache_key)
            if hit is not None:
                return hit

        import soundfile as sf  # type: ignore[import-untyped]

        data, native_sr = sf.read(str(path), dtype="float32", always_2d=False)

        # Mono before resample: see audio_preprocessor._run_processor
        # for the same rationale (mixing after resampling can smear
        # channel-specific transients the filter needs to preserve).
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32, copy=False)
        mono: np.ndarray = np.ascontiguousarray(data, dtype=np.float32)

        if native_sr != self._sample_rate:
            if not self._auto_resample:
                raise ValueError(
                    f"AudioLmCollator: audio {path.name!r} native sample_rate="
                    f"{native_sr} Hz != pinned {self._sample_rate} Hz. "
                    "Set `training.audio.auto_resample: true` to resample "
                    "on the fly, or re-encode with "
                    f"`ffmpeg -i <in> -ar {self._sample_rate} <out>`."
                )
            from dlm.data.audio_resample import resample

            mono = resample(mono, src_sr=native_sr, dst_sr=self._sample_rate)

        max_samples = int(round(self._max_length_seconds * self._sample_rate))
        if mono.shape[0] > max_samples:
            mono = mono[:max_samples]

        if cache_key is not None and self._waveform_cache is not None:
            self._waveform_cache.put(cache_key, mono)

        return mono
