"""Adversarial replay harvest — pull mode.

Post-training, `dlm harvest` reads a sway JSON report, extracts
failing probes with known references, and writes them back as
`!probe`-tagged `::instruction::` sections. The document grows to
contain its own weaknesses; the next retrain picks them up via the
existing probe-sampling path.

Public surface:

- :class:`HarvestCandidate` — one failing probe with its
  reference answer, ready to be materialized as a Section.
- :func:`read_sway_report` — parse a sway JSON file into
  candidates.

Writer + differ land in ``dlm.harvest.diff`` and ``dlm.harvest.applier``
in the next sprint tick.
"""

from __future__ import annotations

from dlm.harvest.errors import (
    HarvestError,
    MalformedSwayReportError,
    NoReferenceError,
)
from dlm.harvest.sway_reader import HarvestCandidate, read_sway_report

__all__ = [
    "HarvestCandidate",
    "HarvestError",
    "MalformedSwayReportError",
    "NoReferenceError",
    "read_sway_report",
]
