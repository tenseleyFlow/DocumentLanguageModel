"""Custom data collator for audio-language training (Sprint 35.2 T8).

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
  whether to decode per-batch or hit the cache. v1 decodes per-batch
  via `soundfile` + hands the waveform to the HF processor; the cache
  from `preprocess_audio` applies to the standalone inference path and
  the slow integration test exercises both.
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
        Rows whose native rate disagrees raise — we refuse silent
        resampling the same way `preprocess_audio` does.
    max_length_seconds:
        Per-clip duration cap in seconds (from
        `AudioPreprocessorPlan`). Longer waveforms are truncated.
    max_length:
        Optional token-length cap for the text side (post-expansion).
        `None` uses the processor's built-in limit.
    """

    def __init__(
        self,
        *,
        processor: Any,
        sample_rate: int,
        max_length_seconds: float,
        max_length: int | None = None,
    ) -> None:
        self._processor = processor
        self._sample_rate = sample_rate
        self._max_length_seconds = max_length_seconds
        self._max_length = max_length

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
            waveforms.append(self._load_waveform(Path(row["audio_path"])))
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

    def _load_waveform(self, path: Path) -> np.ndarray:
        """Decode one audio blob into a mono float32 waveform.

        Refuses on sample-rate mismatch (same gate as
        `preprocess_audio`). Truncates to the configured duration.
        Stereo-to-mono by channel averaging.
        """
        import soundfile as sf  # type: ignore[import-untyped]

        data, native_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if native_sr != self._sample_rate:
            raise ValueError(
                f"AudioLmCollator: audio {path.name!r} native sample_rate="
                f"{native_sr} Hz != pinned {self._sample_rate} Hz; "
                f"re-encode with `ffmpeg -i <in> -ar {self._sample_rate} <out>`"
            )
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32, copy=False)
        mono: np.ndarray = np.ascontiguousarray(data, dtype=np.float32)

        max_samples = int(round(self._max_length_seconds * self._sample_rate))
        if mono.shape[0] > max_samples:
            mono = mono[:max_samples]
        return mono
