"""Preference-mining backend: prompt extraction, sampling, and ranking.

This module stays side-effect-free. It derives prompts from a parsed
`.dlm`, samples candidate responses through a loaded inference backend,
scores candidate pairs with a `PreferenceJudge`, and materializes the
auto-mined `::preference::` sections that a later CLI layer can review
or apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from dlm.data.errors import InstructionParseError
from dlm.data.instruction_parser import parse_instruction_body
from dlm.data.sections_to_rows import _normalize_probe_markers
from dlm.doc.parser import ParsedDlm
from dlm.doc.sections import Section, SectionType
from dlm.preference.judge import PairScore, PreferenceJudge

_NEAR_DUP_RATIO = 0.95


@runtime_checkable
class PreferenceMiningBackend(Protocol):
    """Generation surface the mine loop needs.

    Matches the existing inference backends closely enough that the CLI
    can pass a loaded backend object directly later.
    """

    def generate(self, prompt: str, **gen_kwargs: Any) -> str:
        """Return one candidate completion for `prompt`."""


class PreferenceMineSkipReason(StrEnum):
    """Why a prompt did not yield an auto-mined pair."""

    MALFORMED_INSTRUCTION = "malformed_instruction"
    DUPLICATE_PROMPT = "duplicate_prompt"
    INSUFFICIENT_VARIETY = "insufficient_variety"
    TOO_SIMILAR = "too_similar"
    BELOW_THRESHOLD = "below_threshold"
    ALREADY_PRESENT = "already_present"


@dataclass(frozen=True)
class PreferenceMinePrompt:
    """One mined prompt extracted from an instruction section."""

    section_id: str
    prompt: str


@dataclass(frozen=True)
class PlannedMinedPreference:
    """One auto-mined preference section ready for review/apply."""

    source: PreferenceMinePrompt
    section: Section
    pair_score: PairScore
    candidate_count: int
    chosen: str
    rejected: str


@dataclass(frozen=True)
class SkippedMinePrompt:
    """One source prompt or section that did not produce a mined pair."""

    section_id: str
    prompt: str | None
    reason: PreferenceMineSkipReason
    detail: str = ""


@dataclass(frozen=True)
class PreferenceMinePlan:
    """What the mine loop produced and what it declined to write."""

    additions: tuple[PlannedMinedPreference, ...]
    skipped: tuple[SkippedMinePrompt, ...]


def build_mine_plan(
    parsed: ParsedDlm,
    backend: PreferenceMiningBackend,
    judge: PreferenceJudge,
    *,
    mined_run_id: int,
    samples: int = 4,
    max_pairs: int | None = None,
    threshold: float | None = None,
    temperature: float = 0.7,
    top_p: float | None = None,
    max_new_tokens: int = 256,
    mined_at: str | None = None,
) -> PreferenceMinePlan:
    """Sample/scored mined preference sections from `parsed`.

    The returned sections are not written automatically. Callers review
    or render the plan, then pass the materialized sections through the
    apply backend in a follow-up step.
    """
    if samples < 2:
        raise ValueError(f"samples must be >= 2, got {samples}")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError(f"max_pairs must be >= 1 when set, got {max_pairs}")
    if threshold is not None and threshold < 0.0:
        raise ValueError(f"threshold must be >= 0.0, got {threshold}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

    effective_threshold = judge.suggested_threshold if threshold is None else threshold
    effective_mined_at = _timestamp_now() if mined_at is None else mined_at

    additions: list[PlannedMinedPreference] = []
    skipped: list[SkippedMinePrompt] = []
    existing_ids = {section.section_id for section in parsed.sections}

    for extracted in _extract_prompts(parsed.sections):
        if isinstance(extracted, SkippedMinePrompt):
            skipped.append(extracted)
            continue
        if max_pairs is not None and len(additions) >= max_pairs:
            break

        source = extracted
        candidates = _sample_candidates(
            backend,
            source.prompt,
            samples=samples,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        unique_candidates = _unique_nonempty(candidates)
        if len(unique_candidates) < 2:
            skipped.append(
                SkippedMinePrompt(
                    section_id=source.section_id,
                    prompt=source.prompt,
                    reason=PreferenceMineSkipReason.INSUFFICIENT_VARIETY,
                    detail=(
                        f"need at least 2 unique non-empty candidates, got {len(unique_candidates)}"
                    ),
                )
            )
            continue

        best = _best_pair(source.prompt, unique_candidates, judge=judge)
        if best is None:
            skipped.append(
                SkippedMinePrompt(
                    section_id=source.section_id,
                    prompt=source.prompt,
                    reason=PreferenceMineSkipReason.TOO_SIMILAR,
                    detail=f"all candidate pairs exceeded similarity ratio {_NEAR_DUP_RATIO:.2f}",
                )
            )
            continue
        if best.margin < effective_threshold:
            skipped.append(
                SkippedMinePrompt(
                    section_id=source.section_id,
                    prompt=source.prompt,
                    reason=PreferenceMineSkipReason.BELOW_THRESHOLD,
                    detail=(f"best margin {best.margin:.3f} < threshold {effective_threshold:.3f}"),
                )
            )
            continue

        section = Section(
            type=SectionType.PREFERENCE,
            content=_render_preference_body(
                prompt=source.prompt,
                chosen=best.chosen,
                rejected=best.rejected,
            ),
            auto_mined=True,
            judge_name=judge.name,
            judge_score_chosen=best.score_chosen,
            judge_score_rejected=best.score_rejected,
            mined_at=effective_mined_at,
            mined_run_id=mined_run_id,
        )
        if section.section_id in existing_ids:
            skipped.append(
                SkippedMinePrompt(
                    section_id=source.section_id,
                    prompt=source.prompt,
                    reason=PreferenceMineSkipReason.ALREADY_PRESENT,
                    detail=f"section_id {section.section_id} already in document",
                )
            )
            continue

        additions.append(
            PlannedMinedPreference(
                source=source,
                section=section,
                pair_score=best.raw_score,
                candidate_count=len(unique_candidates),
                chosen=best.chosen,
                rejected=best.rejected,
            )
        )
        existing_ids.add(section.section_id)

    return PreferenceMinePlan(additions=tuple(additions), skipped=tuple(skipped))


def render_mine_plan(plan: PreferenceMinePlan) -> str:
    """Plain-text rendering for dry-run CLI output and tests."""
    lines = [
        f"preference mine plan: {len(plan.additions)} add, {len(plan.skipped)} skip",
        "",
    ]
    if plan.additions:
        lines.append("=== additions ===")
        for add in plan.additions:
            lines.append("")
            lines.append(
                "+ ::preference:: "
                f"[section_id={add.section.section_id} judge={add.section.judge_name} "
                f"margin={abs(add.pair_score.margin):.3f}]"
            )
            lines.append("  prompt: " + _first_line(add.source.prompt))
            lines.append("  chosen: " + _first_line(add.chosen))
            lines.append("  rejected: " + _first_line(add.rejected))
    if plan.skipped:
        lines.append("")
        lines.append("=== skipped ===")
        for skip in plan.skipped:
            label = skip.prompt if skip.prompt else skip.section_id
            lines.append(f"- {label}: {skip.reason.value} ({skip.detail})")
    return "\n".join(lines)


@dataclass(frozen=True)
class _ResolvedPair:
    """A judged pair oriented as chosen vs rejected."""

    chosen: str
    rejected: str
    score_chosen: float
    score_rejected: float
    raw_score: PairScore

    @property
    def margin(self) -> float:
        return self.score_chosen - self.score_rejected


def _extract_prompts(
    sections: tuple[Section, ...] | list[Section],
) -> list[PreferenceMinePrompt | SkippedMinePrompt]:
    out: list[PreferenceMinePrompt | SkippedMinePrompt] = []
    seen_prompts: set[str] = set()
    for section in sections:
        if section.type is not SectionType.INSTRUCTION:
            continue
        try:
            pairs = parse_instruction_body(
                _normalize_probe_markers(section.content),
                section_id=section.section_id,
            )
        except InstructionParseError as exc:
            out.append(
                SkippedMinePrompt(
                    section_id=section.section_id,
                    prompt=None,
                    reason=PreferenceMineSkipReason.MALFORMED_INSTRUCTION,
                    detail=str(exc),
                )
            )
            continue
        for pair in pairs:
            prompt = pair.question.strip()
            if prompt in seen_prompts:
                out.append(
                    SkippedMinePrompt(
                        section_id=section.section_id,
                        prompt=prompt,
                        reason=PreferenceMineSkipReason.DUPLICATE_PROMPT,
                        detail="prompt text already mined from an earlier instruction turn",
                    )
                )
                continue
            seen_prompts.add(prompt)
            out.append(PreferenceMinePrompt(section_id=section.section_id, prompt=prompt))
    return out


def _sample_candidates(
    backend: PreferenceMiningBackend,
    prompt: str,
    *,
    samples: int,
    temperature: float,
    top_p: float | None,
    max_new_tokens: int,
) -> list[str]:
    return [
        str(
            backend.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )
        for _ in range(samples)
    ]


def _unique_nonempty(candidates: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = candidate.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _best_pair(
    prompt: str,
    candidates: list[str],
    *,
    judge: PreferenceJudge,
) -> _ResolvedPair | None:
    best: _ResolvedPair | None = None
    for idx, candidate_a in enumerate(candidates):
        for candidate_b in candidates[idx + 1 :]:
            if _similarity_ratio(candidate_a, candidate_b) > _NEAR_DUP_RATIO:
                continue
            pair_score = judge.score_pair(prompt, candidate_a, candidate_b)
            resolved = _resolve_pair(candidate_a, candidate_b, pair_score)
            if resolved is None:
                continue
            if best is None or resolved.margin > best.margin:
                best = resolved
    return best


def _resolve_pair(
    candidate_a: str,
    candidate_b: str,
    pair_score: PairScore,
) -> _ResolvedPair | None:
    if pair_score.preferred == "tie":
        return None
    if pair_score.preferred == "a":
        return _ResolvedPair(
            chosen=candidate_a,
            rejected=candidate_b,
            score_chosen=pair_score.score_a,
            score_rejected=pair_score.score_b,
            raw_score=pair_score,
        )
    return _ResolvedPair(
        chosen=candidate_b,
        rejected=candidate_a,
        score_chosen=pair_score.score_b,
        score_rejected=pair_score.score_a,
        raw_score=pair_score,
    )


def _render_preference_body(*, prompt: str, chosen: str, rejected: str) -> str:
    return "\n".join(
        [
            "### Prompt",
            prompt,
            "### Chosen",
            chosen,
            "### Rejected",
            rejected,
        ]
    )


def _similarity_ratio(left: str, right: str) -> float:
    return SequenceMatcher(a=left, b=right).ratio()


def _timestamp_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _first_line(text: str, *, max_chars: int = 80) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) > max_chars:
        return first[: max_chars - 1] + "…"
    return first
