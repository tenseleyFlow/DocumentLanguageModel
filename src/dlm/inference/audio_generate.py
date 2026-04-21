"""Audio-language generation path for `dlm prompt --audio`.

Parallel to `vl_generate.py`. Drives an HF `AutoProcessor` + audio-LM
model class (e.g. `Qwen2AudioForConditionalGeneration`) through a
prompt that carries one or more audio placeholders.

Shape contract matches training: the user's prompt carries the base's
`audio_token` placeholder (e.g. `<|AUDIO|>`) and the processor expands
each occurrence into its fixed audio-token window derived from the
waveform's feature count. Our pinned `max_length_seconds` keeps that
count stable across train + inference.

Heavy imports (`soundfile`, `torch`) defer inside the functions so
importing this module stays cheap for non-audio CLI paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dlm.inference.generate import DEFAULT_MAX_NEW_TOKENS, build_generate_kwargs


def format_audio_prompt(
    prompt: str,
    *,
    audio_token: str,
    num_audios: int,
) -> str:
    """Build the audio-aware prompt text.

    Mirrors `format_vl_prompt`. When the user's prompt already carries
    the token, honor their placement; otherwise prepend one token per
    audio so the processor can slot the waveform features in before
    the user's question.
    """
    if audio_token in prompt:
        return prompt
    tokens = audio_token * num_audios
    return f"{tokens}\n{prompt}" if prompt else tokens


def load_audios(paths: list[Path], *, target_sample_rate: int) -> list[np.ndarray]:
    """Open each audio path as a mono float32 waveform at `target_sample_rate`.

    Refuses on sample-rate mismatch (same policy as `preprocess_audio`
    and `AudioLmCollator`). Downmixes stereo to mono by channel
    averaging. `FileNotFoundError` bubbles up from soundfile; the CLI
    converts it to a typer exit.
    """
    import soundfile as sf  # type: ignore[import-untyped]

    waveforms: list[np.ndarray] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"audio not found: {path}")
        data, native_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if native_sr != target_sample_rate:
            raise ValueError(
                f"audio {path.name!r}: native sample_rate={native_sr} Hz "
                f"does not match pinned {target_sample_rate} Hz "
                f"(re-encode with `ffmpeg -i <in> -ar {target_sample_rate} <out>`)"
            )
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32, copy=False)
        waveforms.append(np.ascontiguousarray(data, dtype=np.float32))
    return waveforms


def generate_audio(  # pragma: no cover
    model: Any,
    processor: Any,
    prompt: str,
    audios: list[np.ndarray],
    *,
    audio_token: str,
    sample_rate: int,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
) -> str:
    """Render audio prompt, run generation, decode response-only tokens.

    Pragma'd from unit coverage because it calls `model.generate` on a
    real HF audio-LM model; exercised by the slow audio integration
    test (T12).
    """
    import torch

    formatted = format_audio_prompt(prompt, audio_token=audio_token, num_audios=len(audios))
    inputs = processor(
        audios=audios,
        text=formatted,
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).to(model.device)
    input_len = int(inputs["input_ids"].shape[-1])

    gen_kwargs = build_generate_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    tokenizer = getattr(processor, "tokenizer", processor)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **gen_kwargs,
            pad_token_id=tokenizer.pad_token_id,
        )

    response_tokens = output[0, input_len:]
    decoded = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return str(decoded)
