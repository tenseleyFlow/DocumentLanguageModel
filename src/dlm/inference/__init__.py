"""Inference path for `dlm prompt`.

Heavy imports (`transformers`, `peft`, `torch`) are deferred.
"""

from __future__ import annotations

from dlm.inference.audio_generate import format_audio_prompt, generate_audio, load_audios
from dlm.inference.audio_loader import LoadedAudioInference, load_for_audio_inference
from dlm.inference.errors import (
    AdapterNotFoundError,
    InferenceError,
    InferencePlanError,
)
from dlm.inference.generate import (
    DEFAULT_MAX_NEW_TOKENS,
    build_generate_kwargs,
    format_chat_prompt,
    generate,
)
from dlm.inference.loader import LoadedInference, build_load_kwargs, load_for_inference
from dlm.inference.plan import AttnImpl, InferencePlan, PrecisionLit, resolve_inference
from dlm.inference.vl_generate import format_vl_prompt, generate_vl, load_images
from dlm.inference.vl_loader import LoadedVlInference, load_for_vl_inference

__all__ = [
    "AdapterNotFoundError",
    "AttnImpl",
    "DEFAULT_MAX_NEW_TOKENS",
    "InferenceError",
    "InferencePlan",
    "InferencePlanError",
    "LoadedAudioInference",
    "LoadedInference",
    "LoadedVlInference",
    "PrecisionLit",
    "build_generate_kwargs",
    "build_load_kwargs",
    "format_audio_prompt",
    "format_chat_prompt",
    "format_vl_prompt",
    "generate",
    "generate_audio",
    "generate_vl",
    "load_audios",
    "load_for_audio_inference",
    "load_for_inference",
    "load_for_vl_inference",
    "load_images",
    "resolve_inference",
]
