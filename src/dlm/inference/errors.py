"""Typed errors for the inference path."""

from __future__ import annotations


class InferenceError(Exception):
    """Base for `dlm.inference` errors."""


class AdapterNotFoundError(InferenceError):
    """`store.adapter_current` returned None or points at a missing directory."""


class InferencePlanError(InferenceError):
    """`resolve_inference()` couldn't produce a viable plan for this host."""
