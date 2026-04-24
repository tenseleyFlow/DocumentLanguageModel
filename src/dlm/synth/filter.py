"""Filter raw synth plans: dedup, optional judge scoring, thresholding."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Literal

from dlm.preference import PairScore, PreferenceJudge
from dlm.synth.run import PlannedSynthInstruction, SynthRunPlan

SynthFilterKind = Literal["none", "dedup-only", "sway"]
_NEAR_DUP_RATIO = 0.95
_SWAY_BASELINE_ANSWER = "I don't know."


class SynthFilterSkipReason(StrEnum):
    """Why a raw synth addition did not survive the filter pipeline."""

    DUPLICATE_PAIR = "duplicate_pair"
    JUDGE_DISFAVORED = "judge_disfavored"
    BELOW_THRESHOLD = "below_threshold"


@dataclass(frozen=True)
class FilteredSynthInstruction:
    """One synth addition that survived filtering."""

    addition: PlannedSynthInstruction
    judge_score: PairScore | None = None


@dataclass(frozen=True)
class FilteredSynthSection:
    """One synth addition rejected by a filter stage."""

    addition: PlannedSynthInstruction
    reason: SynthFilterSkipReason
    detail: str = ""
    judge_score: PairScore | None = None


@dataclass(frozen=True)
class SynthFilterReport:
    """Step-by-step counters for the synth filter pipeline."""

    filter_kind: SynthFilterKind
    generated_count: int
    dedup_count: int
    judge_pass_count: int
    accepted_count: int


@dataclass(frozen=True)
class FilteredSynthPlan:
    """Raw synth plan plus the filter pipeline outcome."""

    additions: tuple[FilteredSynthInstruction, ...]
    raw_skipped: tuple[object, ...]
    filtered_skipped: tuple[FilteredSynthSection, ...]
    report: SynthFilterReport


def filter_synth_plan(
    plan: SynthRunPlan,
    *,
    filter_kind: SynthFilterKind = "sway",
    judge: PreferenceJudge | None = None,
    threshold: float | None = None,
) -> FilteredSynthPlan:
    """Apply dedup and optional judge-based filtering to `plan`."""
    if threshold is not None and threshold < 0.0:
        raise ValueError(f"threshold must be >= 0.0, got {threshold}")
    if filter_kind == "sway" and judge is None:
        raise ValueError("filter_kind='sway' requires a judge")

    deduped, dedup_skips = _dedupe(plan.additions)
    filtered_skipped: list[FilteredSynthSection] = list(dedup_skips)
    accepted: list[FilteredSynthInstruction] = []
    judge_pass_count = 0

    if filter_kind in ("none", "dedup-only"):
        accepted = [FilteredSynthInstruction(addition=addition) for addition in deduped]
        judge_pass_count = len(deduped)
    else:
        assert judge is not None
        effective_threshold = judge.suggested_threshold if threshold is None else threshold
        for addition in deduped:
            pair_score = judge.score_pair(
                addition.pair.question,
                addition.pair.answer,
                _SWAY_BASELINE_ANSWER,
            )
            if pair_score.preferred != "a":
                filtered_skipped.append(
                    FilteredSynthSection(
                        addition=addition,
                        reason=SynthFilterSkipReason.JUDGE_DISFAVORED,
                        detail="judge did not prefer the generated answer over the empty baseline",
                        judge_score=pair_score,
                    )
                )
                continue
            judge_pass_count += 1
            if pair_score.margin < effective_threshold:
                filtered_skipped.append(
                    FilteredSynthSection(
                        addition=addition,
                        reason=SynthFilterSkipReason.BELOW_THRESHOLD,
                        detail=(
                            f"margin {pair_score.margin:.3f} < threshold {effective_threshold:.3f}"
                        ),
                        judge_score=pair_score,
                    )
                )
                continue
            accepted.append(FilteredSynthInstruction(addition=addition, judge_score=pair_score))

    return FilteredSynthPlan(
        additions=tuple(accepted),
        raw_skipped=tuple(plan.skipped),
        filtered_skipped=tuple(filtered_skipped),
        report=SynthFilterReport(
            filter_kind=filter_kind,
            generated_count=len(plan.additions),
            dedup_count=len(deduped),
            judge_pass_count=judge_pass_count,
            accepted_count=len(accepted),
        ),
    )


def render_filter_report(plan: FilteredSynthPlan) -> str:
    """Plain-text summary for CLI dry-run output and tests."""
    report = plan.report
    if report.filter_kind == "sway":
        lines = [
            "synth filter: "
            f"generated {report.generated_count}, dedup {report.dedup_count}, "
            f"judge passed {report.judge_pass_count}, threshold {report.accepted_count}",
        ]
    else:
        lines = [
            "synth filter: "
            f"generated {report.generated_count}, dedup {report.dedup_count}, "
            f"accepted {report.accepted_count}",
        ]
    if plan.filtered_skipped:
        lines.append("")
        lines.append("=== filtered ===")
        for skip in plan.filtered_skipped:
            lines.append(
                f"- {skip.addition.section.section_id}: {skip.reason.value} ({skip.detail})"
            )
    return "\n".join(lines)


def _dedupe(
    additions: tuple[PlannedSynthInstruction, ...],
) -> tuple[list[PlannedSynthInstruction], list[FilteredSynthSection]]:
    survivors: list[PlannedSynthInstruction] = []
    skipped: list[FilteredSynthSection] = []

    for addition in additions:
        duplicate_of = _find_duplicate(addition, survivors)
        if duplicate_of is None:
            survivors.append(addition)
            continue
        skipped.append(
            FilteredSynthSection(
                addition=addition,
                reason=SynthFilterSkipReason.DUPLICATE_PAIR,
                detail=f"near-duplicate of section_id {duplicate_of.section.section_id}",
            )
        )
    return survivors, skipped


def _find_duplicate(
    candidate: PlannedSynthInstruction,
    survivors: list[PlannedSynthInstruction],
) -> PlannedSynthInstruction | None:
    question = _normalize_text(candidate.pair.question)
    joined = _dedup_key(candidate)
    for existing in survivors:
        if question == _normalize_text(existing.pair.question):
            return existing
        if _similarity_ratio(joined, _dedup_key(existing)) > _NEAR_DUP_RATIO:
            return existing
    return None


def _dedup_key(addition: PlannedSynthInstruction) -> str:
    return _normalize_text(addition.pair.question) + "\n" + _normalize_text(addition.pair.answer)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _similarity_ratio(left: str, right: str) -> float:
    return SequenceMatcher(a=left, b=right).ratio()
