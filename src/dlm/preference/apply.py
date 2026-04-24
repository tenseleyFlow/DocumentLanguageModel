"""Apply/revert auto-mined preference sections.

This mirrors the harvest write path: build a side-effect-free plan,
then atomically apply it to a `.dlm`. The plan surface keeps the later
CLI thin and gives tests one frozen substrate to assert against.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from dlm.doc.parser import ParsedDlm
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import serialize
from dlm.io.atomic import write_text as atomic_write_text


class PreferenceSkipReason(StrEnum):
    """Why a mined-preference section did not make it into the plan."""

    ALREADY_PRESENT = "already_present"
    NOT_PREFERENCE = "not_preference"
    NOT_AUTO_MINED = "not_auto_mined"


@dataclass(frozen=True)
class PlannedPreferenceAddition:
    """One section that survived plan-time validation and dedupe."""

    section: Section


@dataclass(frozen=True)
class SkippedPreferenceSection:
    """One input section that did not make it into the plan."""

    section: Section
    reason: PreferenceSkipReason
    detail: str = ""


@dataclass(frozen=True)
class PreferenceApplyPlan:
    """What the applier would do if executed."""

    additions: tuple[PlannedPreferenceAddition, ...]
    skipped: tuple[SkippedPreferenceSection, ...]


@dataclass(frozen=True)
class PreferenceApplySummary:
    """Outcome of applying or reverting auto-mined preference sections."""

    target: Path
    added: int
    skipped: int
    added_section_ids: tuple[str, ...]


def build_apply_plan(parsed: ParsedDlm, sections: list[Section]) -> PreferenceApplyPlan:
    """Validate + dedupe mined preference sections against `parsed`."""
    existing = {section.section_id for section in parsed.sections}
    additions: list[PlannedPreferenceAddition] = []
    skipped: list[SkippedPreferenceSection] = []

    for section in sections:
        if section.type is not SectionType.PREFERENCE:
            skipped.append(
                SkippedPreferenceSection(
                    section=section,
                    reason=PreferenceSkipReason.NOT_PREFERENCE,
                    detail="only preference sections can be applied",
                )
            )
            continue
        if not section.auto_mined:
            skipped.append(
                SkippedPreferenceSection(
                    section=section,
                    reason=PreferenceSkipReason.NOT_AUTO_MINED,
                    detail="section is not marked auto_mined=true",
                )
            )
            continue
        if section.section_id in existing:
            skipped.append(
                SkippedPreferenceSection(
                    section=section,
                    reason=PreferenceSkipReason.ALREADY_PRESENT,
                    detail=f"section_id {section.section_id} already in document",
                )
            )
            continue
        additions.append(PlannedPreferenceAddition(section=section))
        existing.add(section.section_id)

    return PreferenceApplyPlan(additions=tuple(additions), skipped=tuple(skipped))


def render_apply_plan(plan: PreferenceApplyPlan) -> str:
    """Plain-text form for dry-run output and tests."""
    lines = [
        f"preference apply plan: {len(plan.additions)} add, {len(plan.skipped)} skip",
        "",
    ]
    if plan.additions:
        lines.append("=== additions ===")
        for add in plan.additions:
            lines.append("")
            lines.append(
                "+ ::preference:: "
                f"[section_id={add.section.section_id} judge={add.section.judge_name}]"
            )
    if plan.skipped:
        lines.append("")
        lines.append("=== skipped ===")
        for skip in plan.skipped:
            lines.append(f"- {skip.section.section_id}: {skip.reason.value} ({skip.detail})")
    return "\n".join(lines)


def apply_plan(
    parsed: ParsedDlm,
    plan: PreferenceApplyPlan,
    *,
    target: Path,
) -> PreferenceApplySummary:
    """Append plan additions to `parsed.sections` and atomically write them."""
    new_sections = tuple(parsed.sections) + tuple(add.section for add in plan.additions)
    updated = dataclasses.replace(parsed, sections=new_sections)
    atomic_write_text(target, serialize(updated))
    return PreferenceApplySummary(
        target=target,
        added=len(plan.additions),
        skipped=len(plan.skipped),
        added_section_ids=tuple(add.section.section_id for add in plan.additions),
    )


def revert_all_auto_mined(
    parsed: ParsedDlm,
    *,
    target: Path,
) -> PreferenceApplySummary:
    """Strip every auto-mined preference section and atomically rewrite `target`."""
    survivors = tuple(
        section
        for section in parsed.sections
        if not (section.type is SectionType.PREFERENCE and section.auto_mined)
    )
    removed_ids = tuple(
        section.section_id
        for section in parsed.sections
        if section.type is SectionType.PREFERENCE and section.auto_mined
    )
    updated = dataclasses.replace(parsed, sections=survivors)
    atomic_write_text(target, serialize(updated))
    return PreferenceApplySummary(
        target=target,
        added=0,
        skipped=0,
        added_section_ids=removed_ids,
    )
