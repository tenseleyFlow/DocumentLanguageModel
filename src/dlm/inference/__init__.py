"""Inference path for `dlm prompt`.

Heavy imports (`transformers`, `peft`, `torch`) are deferred.
"""

from __future__ import annotations

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

__all__ = [
    "AdapterNotFoundError",
    "AttnImpl",
    "DEFAULT_MAX_NEW_TOKENS",
    "InferenceError",
    "InferencePlan",
    "InferencePlanError",
    "LoadedInference",
    "PrecisionLit",
    "build_generate_kwargs",
    "build_load_kwargs",
    "format_chat_prompt",
    "generate",
    "load_for_inference",
    "resolve_inference",
]
