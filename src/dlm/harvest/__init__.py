"""Adversarial replay harvest — pull mode.

Post-training, `dlm harvest` reads a sway JSON report, extracts
failing probes with known references, and writes them back as
`!probe`-tagged `::instruction::` sections. The document grows to
contain its own weaknesses; the next retrain picks them up via the
existing probe-sampling path.

Public surface:

- :class:`HarvestCandidate` / :func:`read_sway_report` — pull
  failing probes out of a sway JSON report.
- :func:`build_plan` / :class:`HarvestPlan` — dedup candidates
  against the current document, materialize Sections.
- :func:`render_plan` — plain-text diff for ``--dry-run``.
- :func:`apply_plan` / :func:`revert_last_harvest` — commit the
  plan to disk (or strip auto-harvested sections on revert).
"""

from __future__ import annotations

from dlm.harvest.applier import HarvestSummary, apply_plan, revert_last_harvest
from dlm.harvest.diff import (
    HarvestPlan,
    PlannedAddition,
    SkippedCandidate,
    SkipReason,
    build_plan,
    render_plan,
)
from dlm.harvest.errors import (
    HarvestError,
    MalformedSwayReportError,
    NoReferenceError,
)
from dlm.harvest.sway_reader import HarvestCandidate, read_sway_report

__all__ = [
    "HarvestCandidate",
    "HarvestError",
    "HarvestPlan",
    "HarvestSummary",
    "MalformedSwayReportError",
    "NoReferenceError",
    "PlannedAddition",
    "SkipReason",
    "SkippedCandidate",
    "apply_plan",
    "build_plan",
    "read_sway_report",
    "render_plan",
    "revert_last_harvest",
]
