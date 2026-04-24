"""Typed errors for Sprint 43 synthetic-instruction generation."""

from __future__ import annotations


class SynthError(RuntimeError):
    """Base typed failure for `dlm synth` flows."""


class TeacherUnavailableError(SynthError):
    """A requested teacher backend is unavailable on this host."""


class InvalidTeacherSpecError(SynthError):
    """The teacher selector string is malformed."""


class TeacherInvocationError(SynthError):
    """A teacher backend was reached but returned an unusable result."""
