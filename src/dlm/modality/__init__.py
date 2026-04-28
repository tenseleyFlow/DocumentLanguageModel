"""Modality dispatch package — replaces scattered ``spec.modality ==`` branches.

Public surface:

- :class:`ModalityDispatch` — base class with predicate flags +
  dispatch hooks (``dispatch_export``, ``load_processor``).
- :data:`MODALITIES` — string → instance registry.
- :func:`modality_for` — resolve a spec to its dispatcher.
- :class:`UnknownModalityError` — raised when a spec's modality
  string has no registered dispatcher.

Callers that previously wrote ``if spec.modality == "vision-language"``
now read ``modality_for(spec).accepts_images`` (or one of the other
predicate flags) or call a dispatch method directly. A pregate
grep-gate refuses new scatter — see ``scripts/pregate.sh``.
Text-family tags (`"text"` and `"text-moe"`) share the same dispatcher.
"""

from __future__ import annotations

from dlm.modality.audio import AudioLanguageModality
from dlm.modality.errors import ModalityError, ProcessorContractError, UnknownModalityError
from dlm.modality.registry import ModalityDispatch, TextModality, modality_for
from dlm.modality.vl import VisionLanguageModality
from dlm.modality.vl_contract import ensure_supported_vl_runtime, validate_loaded_vl_processor

MODALITIES: dict[str, ModalityDispatch] = {
    "text": TextModality(),
    "text-moe": TextModality(),
    "vision-language": VisionLanguageModality(),
    "audio-language": AudioLanguageModality(),
}
"""Registry: modality string → dispatcher instance. Ordered by
registration history — future modalities append here and land a
corresponding class under ``dlm.modality``. Text-family aliases share
the base text dispatcher intentionally."""

__all__ = [
    "MODALITIES",
    "AudioLanguageModality",
    "ModalityDispatch",
    "ModalityError",
    "ProcessorContractError",
    "TextModality",
    "UnknownModalityError",
    "VisionLanguageModality",
    "ensure_supported_vl_runtime",
    "modality_for",
    "validate_loaded_vl_processor",
]
