"""Apply a harvest plan — write additions to the target `.dlm`.

The applier is the side-effecting half of the pull path. It takes a
`HarvestPlan` (from :func:`dlm.harvest.diff.build_plan`) and writes
the updated document back atomically. Non-plan state (the manifest,
the store) is untouched; harvest is a document edit, not a training
event. Provenance rides on ``Section.harvest_source`` for per-section
traceability.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from dlm.doc.parser import ParsedDlm
from dlm.doc.serializer import serialize
from dlm.harvest.diff import HarvestPlan
from dlm.io.atomic import write_text as atomic_write_text


@dataclass(frozen=True)
class HarvestSummary:
    """Outcome of :func:`apply_plan` or :func:`revert_last_harvest`."""

    target: Path
    added: int
    skipped: int
    added_section_ids: tuple[str, ...]


def apply_plan(parsed: ParsedDlm, plan: HarvestPlan, *, target: Path) -> HarvestSummary:
    """Append plan additions to `parsed.sections` and write to `target`.

    Atomic write via :func:`dlm.io.atomic.write_text` — on SIGTERM
    mid-write the target is either the old document or the new one,
    never truncated.
    """
    new_sections = tuple(parsed.sections) + tuple(add.section for add in plan.additions)
    updated = dataclasses.replace(parsed, sections=new_sections)
    rendered = serialize(updated)
    atomic_write_text(target, rendered)
    return HarvestSummary(
        target=target,
        added=len(plan.additions),
        skipped=len(plan.skipped),
        added_section_ids=tuple(add.section.section_id for add in plan.additions),
    )


def revert_last_harvest(parsed: ParsedDlm, *, target: Path) -> HarvestSummary:
    """Strip every `auto_harvest=True` section and rewrite `target`.

    Coarser than "undo the last harvest" — any auto-harvested section
    is removed regardless of which harvest run added it. This matches
    the sprint spec's `--revert HEAD` UX: users audit the diff before
    applying, so "undo all auto-edits" is the safe escape hatch.
    """
    survivors = tuple(s for s in parsed.sections if not s.auto_harvest)
    removed_ids = tuple(s.section_id for s in parsed.sections if s.auto_harvest)
    updated = dataclasses.replace(parsed, sections=survivors)
    rendered = serialize(updated)
    atomic_write_text(target, rendered)
    return HarvestSummary(
        target=target,
        added=0,
        skipped=0,
        added_section_ids=removed_ids,
    )
