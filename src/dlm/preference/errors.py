"""Errors raised by the preference-mining infrastructure."""

from __future__ import annotations


class PreferenceMiningError(RuntimeError):
    """Root of the preference-mining exception hierarchy."""


class JudgeUnavailableError(PreferenceMiningError):
    """The requested judge exists in principle but cannot run here."""


class InvalidJudgeSpecError(PreferenceMiningError):
    """A CLI/user judge selector string was malformed."""


class JudgeInvocationError(PreferenceMiningError):
    """A concrete judge ran but returned an unusable result."""
