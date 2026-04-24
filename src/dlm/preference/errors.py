"""Errors raised by Sprint 42 preference-mining infrastructure."""

from __future__ import annotations


class PreferenceMiningError(RuntimeError):
    """Root of the preference-mining exception hierarchy."""


class JudgeUnavailableError(PreferenceMiningError):
    """The requested judge exists in principle but cannot run here."""


class InvalidJudgeSpecError(PreferenceMiningError):
    """A CLI/user judge selector string was malformed."""
