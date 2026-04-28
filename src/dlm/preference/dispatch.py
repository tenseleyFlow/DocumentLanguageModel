"""Domain dispatcher for `dlm preference mine`.

Lifts the build-backend → load → build-judge → mine → stage/apply
pipeline (and metrics record) out of the CLI. Callers (CLI, LSP, future
automation) build a `PreferenceMineRequest`, call `run_preference_mine`,
and render the typed `PreferenceMineResult` themselves. The dispatcher
does no console I/O; backend / judge / mine errors propagate as the
existing typed exceptions so the caller can map each to its own exit
code or banner.

External-module imports are dotted (e.g. `from dlm.inference import
backends as _backends; _backends.build_backend(...)`) so test fixtures
that monkeypatch `dlm.inference.backends.<name>` are visible to the
dispatcher at call time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from dlm.inference import backends as _backends
from dlm.inference.backends.select import BackendName
from dlm.metrics import MetricsRecorder
from dlm.metrics.events import PreferenceMineEvent
from dlm.preference import apply as _apply
from dlm.preference import judge as _judge_mod
from dlm.preference import mine as _mine
from dlm.preference import pending as _pending
from dlm.preference.apply import PreferenceApplyPlan, PreferenceApplySummary
from dlm.preference.mine import PreferenceMinePlan

if TYPE_CHECKING:
    from pathlib import Path

    from dlm.base_models.schema import BaseModelSpec
    from dlm.doc.parser import ParsedDlm
    from dlm.hardware.capabilities import Capabilities
    from dlm.store.paths import StorePath


class PreferenceMineOutcome(StrEnum):
    """Discriminator for what `run_preference_mine` did."""

    APPLIED = "applied"  # Sections written to the .dlm.
    STAGED = "staged"  # Sections persisted as the pending plan.
    NO_ADDITIONS = "no_additions"  # Mining yielded no confident pairs.


@dataclass(frozen=True)
class PreferenceMineRequest:
    """Inputs to `run_preference_mine`.

    The CLI is responsible for parsing the .dlm, resolving the store,
    enforcing license acceptance, and selecting the backend kind; the
    dispatcher receives all of those as already-typed objects.
    `mined_run_id` is required — the CLI exits early when no prior
    training run exists.
    """

    parsed: ParsedDlm
    target_path: Path
    store: StorePath
    spec: BaseModelSpec
    capabilities: Capabilities
    backend_name: BackendName
    judge_spec: str
    mined_run_id: int
    samples: int
    max_pairs: int | None
    threshold: float | None
    temperature: float
    top_p: float | None
    adapter: str | None
    apply: bool


@dataclass(frozen=True)
class PreferenceMineResult:
    """Outcome of `run_preference_mine`.

    `plan` is always populated so the CLI can render it. `apply_plan`
    and `apply_summary` are only set on `PreferenceMineOutcome.APPLIED`.
    `pending_count` is non-zero only on `PreferenceMineOutcome.STAGED`.
    `judge_name` is the resolved judge identifier, used by the CLI for
    metrics narration.
    """

    plan: PreferenceMinePlan
    outcome: PreferenceMineOutcome
    judge_name: str
    apply_plan: PreferenceApplyPlan | None = None
    apply_summary: PreferenceApplySummary | None = None
    pending_count: int = 0


def run_preference_mine(req: PreferenceMineRequest) -> PreferenceMineResult:
    """Build, load, mine, and stage/apply preference sections for one .dlm."""
    backend_obj = _backends.build_backend(req.backend_name, req.capabilities)
    backend_obj.load(req.spec, req.store, adapter_name=req.adapter)

    try:
        judge_obj = _judge_mod.build_judge(req.judge_spec, dlm_path=req.target_path)
        judge_name = judge_obj.name
        plan = _mine.build_mine_plan(
            req.parsed,
            backend_obj,
            judge_obj,
            mined_run_id=req.mined_run_id,
            samples=req.samples,
            max_pairs=req.max_pairs,
            threshold=req.threshold,
            temperature=req.temperature,
            top_p=req.top_p,
        )
    finally:
        backend_obj.unload()

    recorder = MetricsRecorder(req.store.root)

    if not plan.additions:
        _pending.clear_pending_plan(req.store)
        recorder.record_preference_mine(
            PreferenceMineEvent(
                run_id=req.mined_run_id,
                judge_name=judge_name,
                sample_count=req.samples,
                mined_pairs=0,
                skipped_prompts=len(plan.skipped),
                write_mode="empty",
            )
        )
        return PreferenceMineResult(
            plan=plan,
            outcome=PreferenceMineOutcome.NO_ADDITIONS,
            judge_name=judge_name,
        )

    sections = [addition.section for addition in plan.additions]

    if req.apply:
        apply_plan = _apply.build_apply_plan(req.parsed, sections)
        summary = _apply.apply_plan(req.parsed, apply_plan, target=req.target_path)
        _pending.clear_pending_plan(req.store)
        recorder.record_preference_mine(
            PreferenceMineEvent(
                run_id=req.mined_run_id,
                judge_name=judge_name,
                sample_count=req.samples,
                mined_pairs=len(plan.additions),
                skipped_prompts=len(plan.skipped),
                write_mode="applied",
            )
        )
        return PreferenceMineResult(
            plan=plan,
            outcome=PreferenceMineOutcome.APPLIED,
            judge_name=judge_name,
            apply_plan=apply_plan,
            apply_summary=summary,
        )

    pending = _pending.save_pending_plan(
        req.store,
        source_path=req.target_path.resolve(),
        sections=sections,
    )
    recorder.record_preference_mine(
        PreferenceMineEvent(
            run_id=req.mined_run_id,
            judge_name=judge_name,
            sample_count=req.samples,
            mined_pairs=len(plan.additions),
            skipped_prompts=len(plan.skipped),
            write_mode="staged",
        )
    )
    return PreferenceMineResult(
        plan=plan,
        outcome=PreferenceMineOutcome.STAGED,
        judge_name=judge_name,
        pending_count=len(pending.sections),
    )
