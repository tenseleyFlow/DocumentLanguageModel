"""Errors raised by the gate module.

Non-fatal: gate-training failures log + fall back to uniform routing
rather than aborting the surrounding SFT commit. The caller decides
whether to treat a `GateTrainingError` as fatal.
"""

from __future__ import annotations


class GateError(Exception):
    """Base for all gate-side errors."""


class GateTrainingError(GateError):
    """Gate training failed (nan loss, zero samples, diverged)."""


class GateConfigError(GateError):
    """Gate metadata on disk is malformed or references missing adapters."""
