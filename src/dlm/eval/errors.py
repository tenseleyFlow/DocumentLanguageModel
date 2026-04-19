"""Typed errors for the eval harness."""

from __future__ import annotations


class EvalError(Exception):
    """Base for `dlm.eval` errors."""


class ProbeFormatError(EvalError):
    """A probe prompt was malformed (e.g., `!probe` header without body)."""


class RetentionSliceError(EvalError):
    """The retention slice couldn't be built (empty corpus, bad seed, etc.)."""
