"""Typed errors for `dlm harvest`.

These surface through the CLI reporter with the same
``file:line:col`` treatment as other `DlmError` subclasses.
"""

from __future__ import annotations


class HarvestError(Exception):
    """Base exception for the harvest pull path."""


class MalformedSwayReportError(HarvestError):
    """The sway JSON report is unreadable or has an unexpected shape.

    Raised when:
    - The file doesn't parse as JSON.
    - Required top-level keys are missing (``schema_version``,
      ``probes``).
    - ``schema_version`` is newer than this reader supports.
    """


class NoReferenceError(HarvestError):
    """A failing probe has no reference answer we can harvest.

    Harvest refuses probes without references rather than silently
    writing incomplete `!probe` sections. Callers can downgrade this
    to a WARN via ``read_sway_report(..., strict=False)``.
    """
