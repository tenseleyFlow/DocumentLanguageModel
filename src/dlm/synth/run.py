"""Pure dry-run orchestration for Sprint 43 synthetic instruction generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from dlm.doc.parser import ParsedDlm
from dlm.doc.sections import Section, SectionType
from dlm.synth.prompts import PromptParserKind, SynthStrategy, get_prompt_template
from dlm.synth.teachers import SynthTeacher

_NUMBERED_Q_RE = re.compile(r"^\s*\d+\.\s*(?:Q|Question):\s*(.+)\s*$", re.IGNORECASE)
_NUMBERED_A_RE = re.compile(r"^\s*(?:A|Answer):\s*(.+)\s*$", re.IGNORECASE)
_FENCED_JSON_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


class SynthSkipReason(StrEnum):
    """Why one prose section or generated pair did not produce an addition."""

    NO_PROSE = "no_prose"
    INVALID_OUTPUT = "invalid_output"
    EMPTY_PAIR = "empty_pair"
    ALREADY_PRESENT = "already_present"


@dataclass(frozen=True)
class SynthPair:
    """One generated instruction pair before materialization."""

    question: str
    answer: str


@dataclass(frozen=True)
class SynthSourceSection:
    """One prose section selected for synthesis."""

    section_id: str
    content: str


ConcreteSynthStrategy = Literal["extraction", "expansion"]


@dataclass(frozen=True)
class PlannedSynthInstruction:
    """One auto-synth instruction section ready for review/apply."""

    source: SynthSourceSection
    strategy: ConcreteSynthStrategy
    pair: SynthPair
    section: Section


@dataclass(frozen=True)
class SkippedSynthSection:
    """One source section or generated pair that the dry-run declined to add."""

    section_id: str
    strategy: ConcreteSynthStrategy | None
    reason: SynthSkipReason
    detail: str = ""


@dataclass(frozen=True)
class SynthRunPlan:
    """What the synth loop produced and what it declined to add."""

    additions: tuple[PlannedSynthInstruction, ...]
    skipped: tuple[SkippedSynthSection, ...]


def build_synth_plan(
    parsed: ParsedDlm,
    teacher: SynthTeacher,
    *,
    per_section: int = 3,
    strategy: SynthStrategy = "extraction",
    max_pairs: int | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float | None = None,
    seed: int | None = None,
    synth_at: str | None = None,
) -> SynthRunPlan:
    """Generate a dry-run synth plan from prose sections in `parsed`."""
    if per_section < 1:
        raise ValueError(f"per_section must be >= 1, got {per_section}")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError(f"max_pairs must be >= 1 when set, got {max_pairs}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

    additions: list[PlannedSynthInstruction] = []
    skipped: list[SkippedSynthSection] = []
    existing_ids = {section.section_id for section in parsed.sections}
    effective_synth_at = synth_at if synth_at is not None else _timestamp_now()

    sources = _extract_prose_sources(parsed.sections)
    if not sources:
        return SynthRunPlan(
            additions=(),
            skipped=(
                SkippedSynthSection(
                    section_id="(document)",
                    strategy=None,
                    reason=SynthSkipReason.NO_PROSE,
                    detail="document has no non-empty PROSE sections to synthesize from",
                ),
            ),
        )

    for source in sources:
        for concrete_strategy, count in _strategy_counts(strategy, per_section):
            if count == 0:
                continue

            template = get_prompt_template(concrete_strategy)
            rendered = teacher.generate(
                template.system_prompt,
                template.render_user_prompt(prose=source.content, n=count),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            try:
                pairs = _parse_generated_pairs(rendered, parser=template.output_parser)
            except ValueError as exc:
                skipped.append(
                    SkippedSynthSection(
                        section_id=source.section_id,
                        strategy=concrete_strategy,
                        reason=SynthSkipReason.INVALID_OUTPUT,
                        detail=str(exc),
                    )
                )
                continue

            for pair in pairs[:count]:
                materialized = Section(
                    type=SectionType.INSTRUCTION,
                    content=_render_instruction_body(pair),
                    auto_synth=True,
                    synth_teacher=teacher.name,
                    synth_strategy=concrete_strategy,
                    synth_at=effective_synth_at,
                    source_section_id=source.section_id,
                )
                if materialized.section_id in existing_ids:
                    skipped.append(
                        SkippedSynthSection(
                            section_id=source.section_id,
                            strategy=concrete_strategy,
                            reason=SynthSkipReason.ALREADY_PRESENT,
                            detail=f"section_id {materialized.section_id} already in document",
                        )
                    )
                    continue
                additions.append(
                    PlannedSynthInstruction(
                        source=source,
                        strategy=concrete_strategy,
                        pair=pair,
                        section=materialized,
                    )
                )
                existing_ids.add(materialized.section_id)
                if max_pairs is not None and len(additions) >= max_pairs:
                    return SynthRunPlan(additions=tuple(additions), skipped=tuple(skipped))

    return SynthRunPlan(additions=tuple(additions), skipped=tuple(skipped))


def render_synth_plan(plan: SynthRunPlan) -> str:
    """Plain-text rendering for dry-run CLI output and tests."""
    lines = [
        f"synth plan: {len(plan.additions)} add, {len(plan.skipped)} skip",
        "",
    ]
    if plan.additions:
        lines.append("=== additions ===")
        for add in plan.additions:
            lines.append("")
            lines.append(
                "+ ::instruction:: "
                f"[section_id={add.section.section_id} source={add.source.section_id} "
                f"teacher={add.section.synth_teacher} strategy={add.strategy}]"
            )
            lines.append("  q: " + _first_line(add.pair.question))
            lines.append("  a: " + _first_line(add.pair.answer))
    if plan.skipped:
        lines.append("")
        lines.append("=== skipped ===")
        for skip in plan.skipped:
            strategy = skip.strategy if skip.strategy is not None else "-"
            lines.append(f"- {skip.section_id} [{strategy}]: {skip.reason.value} ({skip.detail})")
    return "\n".join(lines)


def _extract_prose_sources(
    sections: tuple[Section, ...] | list[Section],
) -> list[SynthSourceSection]:
    return [
        SynthSourceSection(section_id=section.section_id, content=section.content.strip())
        for section in sections
        if section.type is SectionType.PROSE and section.content.strip()
    ]


def _strategy_counts(
    strategy: SynthStrategy, per_section: int
) -> list[tuple[ConcreteSynthStrategy, int]]:
    if strategy == "extraction":
        return [("extraction", per_section)]
    if strategy == "expansion":
        return [("expansion", per_section)]
    extraction = (per_section + 1) // 2
    expansion = per_section // 2
    return [("extraction", extraction), ("expansion", expansion)]


def _parse_generated_pairs(raw: str, *, parser: PromptParserKind) -> list[SynthPair]:
    pairs = (
        _parse_json_list_pairs(raw) if parser == "json_list" else _parse_numbered_list_pairs(raw)
    )
    if not pairs:
        raise ValueError("teacher output produced no instruction pairs")
    return pairs


def _parse_json_list_pairs(raw: str) -> list[SynthPair]:
    candidate = _strip_json_fence(raw)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"teacher output is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("teacher output must be a JSON list")

    pairs: list[SynthPair] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"teacher output item {idx} must be an object")
        question = item.get("question")
        answer = item.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError(f"teacher output item {idx} must contain string question/answer keys")
        question_text = question.strip()
        answer_text = answer.strip()
        if not question_text or not answer_text:
            raise ValueError(f"teacher output item {idx} has an empty question or answer")
        pairs.append(SynthPair(question=question_text, answer=answer_text))
    return pairs


def _strip_json_fence(raw: str) -> str:
    match = _FENCED_JSON_RE.match(raw)
    if match is None:
        return raw
    return match.group(1).strip()


def _parse_numbered_list_pairs(raw: str) -> list[SynthPair]:
    lines = [line for line in raw.splitlines() if line.strip()]
    pairs: list[SynthPair] = []
    idx = 0
    while idx < len(lines):
        q_match = _NUMBERED_Q_RE.match(lines[idx])
        if q_match is None:
            raise ValueError(
                "teacher output numbered_list must use lines like `1. Q: ...` or `1. Question: ...`"
            )
        idx += 1
        if idx >= len(lines):
            raise ValueError("teacher output numbered_list is missing an answer line")
        a_match = _NUMBERED_A_RE.match(lines[idx])
        if a_match is None:
            raise ValueError(
                "teacher output numbered_list answers must use `A:` or `Answer:` lines"
            )
        idx += 1
        question = q_match.group(1).strip()
        answer = a_match.group(1).strip()
        if not question or not answer:
            raise ValueError("teacher output numbered_list contains an empty question or answer")
        pairs.append(SynthPair(question=question, answer=answer))
    return pairs


def _render_instruction_body(pair: SynthPair) -> str:
    return "\n".join(
        [
            "### Q",
            pair.question,
            "### A",
            pair.answer,
        ]
    )


def _timestamp_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _first_line(text: str, *, max_chars: int = 80) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) > max_chars:
        return first[: max_chars - 1] + "…"
    return first
