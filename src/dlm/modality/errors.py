"""Typed errors for modality dispatch."""

from __future__ import annotations


class ModalityError(Exception):
    """Base for `dlm.modality` errors."""


class UnknownModalityError(ModalityError):
    """Spec declares a modality string the registry doesn't know."""


class ProcessorContractError(ModalityError):
    """The resolved media processor can't satisfy DLM's runtime contract."""
