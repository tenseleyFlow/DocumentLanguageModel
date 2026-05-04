"""Domain dispatcher for `dlm synth instructions`.

Lifts the build-teacher → build-plan → filter → stage/apply pipeline
out of the CLI. Callers (CLI, LSP, future automation) build a
`SynthInstructionsRequest`, call `run_synth_instructions`, and render
the typed `SynthInstructionsResult` themselves. The dispatcher does no
console I/O; teacher/judge/plan errors propagate as the existing typed
exceptions so the caller can map each to its own exit code or banner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from dlm.synth import apply as _apply
from dlm.synth import filter as _filter
from dlm.synth import pending as _pending
from dlm.synth import run as _run
from dlm.synth import teachers as _teachers
from dlm.synth.apply import SynthApplyPlan, SynthApplySummary
from dlm.synth.filter import FilteredSynthPlan, SynthFilterKind
from dlm.synth.prompts import SynthStrategy
from dlm.synth.run import SynthRunPlan

if TYPE_CHECKING:
    from pathlib import Path

    from dlm.doc.parser import ParsedDlm
    from dlm.preference.judge import PreferenceJudge
    from dlm.store.paths import StorePath


class SynthOutcome(StrEnum):
    """Discriminator for what `run_synth_instructions` did."""

    APPLIED = "applied"  # Sections written to the .dlm.
    STAGED = "staged"  # Sections persisted as the pending plan.
    DRY_RUN = "dry_run"  # Plan rendered, nothing staged or written.
    NO_ADDITIONS = "no_additions"  # Filter rejected every candidate.


@dataclass(frozen=True)
class SynthInstructionsRequest:
    """Inputs to `run_synth_instructions`.

    The CLI is responsible for parsing the .dlm and resolving the
    store; the dispatcher receives both as already-typed objects.
    `apply` and `dry_run` are mutually exclusive — the CLI rejects
    that combo before constructing the request.
    """

    parsed: ParsedDlm
    target_path: Path
    store: StorePath
    teacher: str
    per_section: int
    strategy: SynthStrategy
    filter_kind: SynthFilterKind
    threshold: float | None
    max_pairs: int | None
    max_new_tokens: int
    temperature: float
    top_p: float | None
    seed: int | None
    apply: bool
    dry_run: bool


@dataclass(frozen=True)
class SynthInstructionsResult:
    """Outcome of `run_synth_instructions`.

    `plan` and `filtered_plan` are always populated so the CLI can
    render them. `apply_plan` / `apply_summary` are only set on
    `SynthOutcome.APPLIED`. `pending_count` is non-zero only on
    `SynthOutcome.STAGED`.
    """

    plan: SynthRunPlan
    filtered_plan: FilteredSynthPlan
    outcome: SynthOutcome
    apply_plan: SynthApplyPlan | None = None
    apply_summary: SynthApplySummary | None = None
    pending_count: int = 0


def run_synth_instructions(req: SynthInstructionsRequest) -> SynthInstructionsResult:
    """Build, filter, and stage/apply auto-synth instructions for one .dlm."""
    teacher_obj = _teachers.build_teacher(req.teacher, dlm_path=req.target_path)
    plan = _run.build_synth_plan(
        req.parsed,
        teacher_obj,
        per_section=req.per_section,
        strategy=req.strategy,
        max_pairs=req.max_pairs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
    )

    judge_obj: PreferenceJudge | None = None
    if req.filter_kind == "sway":
        from dlm.preference.judge import build_judge

        judge_obj = build_judge("sway", dlm_path=req.target_path)

    filtered = _filter.filter_synth_plan(
        plan,
        filter_kind=req.filter_kind,
        judge=judge_obj,
        threshold=req.threshold,
    )

    if not filtered.additions:
        if not req.dry_run:
            _pending.clear_pending_plan(req.store)
        return SynthInstructionsResult(
            plan=plan,
            filtered_plan=filtered,
            outcome=SynthOutcome.NO_ADDITIONS,
        )

    sections = [addition.addition.section for addition in filtered.additions]

    if req.apply:
        apply_plan = _apply.build_apply_plan(req.parsed, sections)
        summary = _apply.apply_plan(req.parsed, apply_plan, target=req.target_path)
        _pending.clear_pending_plan(req.store)
        return SynthInstructionsResult(
            plan=plan,
            filtered_plan=filtered,
            outcome=SynthOutcome.APPLIED,
            apply_plan=apply_plan,
            apply_summary=summary,
        )

    if req.dry_run:
        return SynthInstructionsResult(
            plan=plan,
            filtered_plan=filtered,
            outcome=SynthOutcome.DRY_RUN,
        )

    pending = _pending.save_pending_plan(
        req.store,
        source_path=req.target_path.resolve(),
        sections=sections,
    )
    return SynthInstructionsResult(
        plan=plan,
        filtered_plan=filtered,
        outcome=SynthOutcome.STAGED,
        pending_count=len(pending.sections),
    )
