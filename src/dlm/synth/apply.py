"""Apply/revert staged auto-synth instruction sections."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from dlm.doc.parser import ParsedDlm
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import serialize
from dlm.io.atomic import write_text as atomic_write_text


class SynthApplySkipReason(StrEnum):
    """Why a staged synth section did not make it into the plan."""

    ALREADY_PRESENT = "already_present"
    NOT_INSTRUCTION = "not_instruction"
    NOT_AUTO_SYNTH = "not_auto_synth"


@dataclass(frozen=True)
class PlannedSynthAddition:
    """One section that survived plan-time validation and dedupe."""

    section: Section


@dataclass(frozen=True)
class SkippedSynthAddition:
    """One input section that did not make it into the plan."""

    section: Section
    reason: SynthApplySkipReason
    detail: str = ""


@dataclass(frozen=True)
class SynthApplyPlan:
    """What the applier would do if executed."""

    additions: tuple[PlannedSynthAddition, ...]
    skipped: tuple[SkippedSynthAddition, ...]


@dataclass(frozen=True)
class SynthApplySummary:
    """Outcome of applying or reverting auto-synth instruction sections."""

    target: Path
    added: int
    skipped: int
    added_section_ids: tuple[str, ...]


def build_apply_plan(parsed: ParsedDlm, sections: list[Section]) -> SynthApplyPlan:
    """Validate + dedupe staged synth sections against `parsed`."""
    existing = {section.section_id for section in parsed.sections}
    additions: list[PlannedSynthAddition] = []
    skipped: list[SkippedSynthAddition] = []

    for section in sections:
        if section.type is not SectionType.INSTRUCTION:
            skipped.append(
                SkippedSynthAddition(
                    section=section,
                    reason=SynthApplySkipReason.NOT_INSTRUCTION,
                    detail="only instruction sections can be applied",
                )
            )
            continue
        if not section.auto_synth:
            skipped.append(
                SkippedSynthAddition(
                    section=section,
                    reason=SynthApplySkipReason.NOT_AUTO_SYNTH,
                    detail="section is not marked auto_synth=true",
                )
            )
            continue
        if section.section_id in existing:
            skipped.append(
                SkippedSynthAddition(
                    section=section,
                    reason=SynthApplySkipReason.ALREADY_PRESENT,
                    detail=f"section_id {section.section_id} already in document",
                )
            )
            continue
        additions.append(PlannedSynthAddition(section=section))
        existing.add(section.section_id)

    return SynthApplyPlan(additions=tuple(additions), skipped=tuple(skipped))


def render_apply_plan(plan: SynthApplyPlan) -> str:
    """Plain-text form for dry-run output and tests."""
    lines = [
        f"synth apply plan: {len(plan.additions)} add, {len(plan.skipped)} skip",
        "",
    ]
    if plan.additions:
        lines.append("=== additions ===")
        for add in plan.additions:
            lines.append("")
            lines.append(
                "+ ::instruction:: "
                "[section_id="
                f"{add.section.section_id} teacher={add.section.synth_teacher} "
                f"strategy={add.section.synth_strategy} source={add.section.source_section_id}]"
            )
    if plan.skipped:
        lines.append("")
        lines.append("=== skipped ===")
        for skip in plan.skipped:
            lines.append(f"- {skip.section.section_id}: {skip.reason.value} ({skip.detail})")
    return "\n".join(lines)


def apply_plan(
    parsed: ParsedDlm,
    plan: SynthApplyPlan,
    *,
    target: Path,
) -> SynthApplySummary:
    """Append plan additions to `parsed.sections` and atomically write them."""
    new_sections = tuple(parsed.sections) + tuple(add.section for add in plan.additions)
    updated = dataclasses.replace(parsed, sections=new_sections)
    atomic_write_text(target, serialize(updated))
    return SynthApplySummary(
        target=target,
        added=len(plan.additions),
        skipped=len(plan.skipped),
        added_section_ids=tuple(add.section.section_id for add in plan.additions),
    )


def revert_all_auto_synth(
    parsed: ParsedDlm,
    *,
    target: Path,
) -> SynthApplySummary:
    """Strip every auto-synth instruction section and atomically rewrite `target`."""
    survivors = tuple(
        section
        for section in parsed.sections
        if not (section.type is SectionType.INSTRUCTION and section.auto_synth)
    )
    removed_ids = tuple(
        section.section_id
        for section in parsed.sections
        if section.type is SectionType.INSTRUCTION and section.auto_synth
    )
    updated = dataclasses.replace(parsed, sections=survivors)
    atomic_write_text(target, serialize(updated))
    return SynthApplySummary(
        target=target,
        added=0,
        skipped=0,
        added_section_ids=removed_ids,
    )
