"""Render a harvest plan — which candidates would be added, which skipped.

The diff is what the user sees under ``dlm harvest --dry-run``. Kept
side-effect-free so the same structure feeds both the Rich rendering
and the applier's JSON reporting mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from dlm.doc.parser import ParsedDlm
from dlm.doc.sections import Section, SectionType
from dlm.harvest.sway_reader import HarvestCandidate


class SkipReason(StrEnum):
    """Why a candidate didn't make it into the plan."""

    ALREADY_PRESENT = "already_present"
    """A section with this content-hash already exists in the document."""

    BELOW_CONFIDENCE = "below_confidence"
    """Reader-level filter — kept here for plan completeness even
    though the reader drops these earlier."""


@dataclass(frozen=True)
class PlannedAddition:
    """One candidate that survived plan-time dedup."""

    candidate: HarvestCandidate
    section: Section
    """The Section that would be appended to the document body."""


@dataclass(frozen=True)
class SkippedCandidate:
    """One candidate that didn't make it into the plan."""

    candidate: HarvestCandidate
    reason: SkipReason
    detail: str = ""


@dataclass(frozen=True)
class HarvestPlan:
    """What the applier would do, frozen for review.

    The applier consumes this; the dry-run renderer also consumes it.
    Both paths see the same thing.
    """

    additions: tuple[PlannedAddition, ...]
    skipped: tuple[SkippedCandidate, ...]


def build_plan(
    parsed: ParsedDlm,
    candidates: list[HarvestCandidate],
    *,
    tag: str = "auto-harvest",
) -> HarvestPlan:
    """Materialize candidates into Sections and dedup against `parsed`.

    Parameters
    ----------
    parsed:
        The current `ParsedDlm` the candidates would be written into.
    candidates:
        Harvest candidates from :func:`read_sway_report`.
    tag:
        Value to write into the synthesized section's
        ``harvest_source``. The CLI defaults to ``"auto-harvest"``.
        The probe name is always concatenated for traceability:
        ``"<tag>/<probe_name>"``.
    """
    existing = {s.section_id for s in parsed.sections}
    additions: list[PlannedAddition] = []
    skipped: list[SkippedCandidate] = []

    for candidate in candidates:
        section = _candidate_to_section(candidate, tag=tag)
        if section.section_id in existing:
            skipped.append(
                SkippedCandidate(
                    candidate=candidate,
                    reason=SkipReason.ALREADY_PRESENT,
                    detail=f"section_id {section.section_id} already in document",
                )
            )
            continue
        additions.append(PlannedAddition(candidate=candidate, section=section))
        # Dedup within the plan too — two sway probes with identical
        # (prompt, reference) shouldn't both add.
        existing.add(section.section_id)

    return HarvestPlan(additions=tuple(additions), skipped=tuple(skipped))


def _candidate_to_section(candidate: HarvestCandidate, *, tag: str) -> Section:
    """Build the `Section` that represents this candidate on disk.

    Shape: `::instruction::` with `### Q` / `### A` body, plus
    `auto_harvest=True` and `harvest_source="<tag>/<probe_name>"`.
    The fence body follows the `!probe` convention so the existing
    probe-extraction path (`src/dlm/eval/probes.py`) picks it up on
    the next train.
    """
    content_lines = [
        "!probe",
        "",
        "### Q",
        candidate.prompt.strip(),
        "",
        "### A",
        candidate.reference.strip(),
    ]
    content = "\n".join(content_lines)
    harvest_source = f"{tag}/{candidate.probe_name}" if tag else candidate.probe_name
    return Section(
        type=SectionType.INSTRUCTION,
        content=content,
        auto_harvest=True,
        harvest_source=harvest_source,
    )


def render_plan(plan: HarvestPlan) -> str:
    """Render a plan as a plain-text diff for stdout / logs.

    The Rich console version lives in `dlm.cli.commands` where the
    Console is already configured; this plain-text form is the
    substrate the CLI wraps and the test fixtures snapshot.
    """
    lines: list[str] = []
    lines.append(f"harvest plan: {len(plan.additions)} add, {len(plan.skipped)} skip")
    lines.append("")
    if plan.additions:
        lines.append("=== additions ===")
        for add in plan.additions:
            lines.append("")
            lines.append(
                f"+ ::instruction:: [probe={add.candidate.probe_name} "
                f"confidence={add.candidate.confidence:.2f} "
                f"section_id={add.section.section_id}]"
            )
            lines.append(f'  source: "{add.section.harvest_source}"')
            lines.append("  Q: " + _first_line(add.candidate.prompt))
            lines.append("  A: " + _first_line(add.candidate.reference))
    if plan.skipped:
        lines.append("")
        lines.append("=== skipped ===")
        for skip in plan.skipped:
            lines.append(f"- {skip.candidate.probe_name}: {skip.reason.value} ({skip.detail})")
    return "\n".join(lines)


def _first_line(text: str, *, max_chars: int = 80) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) > max_chars:
        return first[: max_chars - 1] + "…"
    return first
