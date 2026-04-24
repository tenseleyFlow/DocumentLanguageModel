"""Filter pipeline for Sprint 43 synthetic instruction generation."""

from __future__ import annotations

import pytest

from dlm.doc.sections import Section, SectionType
from dlm.preference import PairScore
from dlm.synth import (
    PlannedSynthInstruction,
    SynthPair,
    SynthRunPlan,
    SynthSourceSection,
    filter_synth_plan,
    render_filter_report,
)


def _planned(
    *,
    source_section_id: str = "aaaaaaaaaaaaaaaa",
    question: str = "What is DGEMM?",
    answer: str = "A matrix multiply routine.",
    strategy: str = "extraction",
) -> PlannedSynthInstruction:
    pair = SynthPair(question=question, answer=answer)
    section = Section(
        type=SectionType.INSTRUCTION,
        content=f"### Q\n{question}\n### A\n{answer}",
        auto_synth=True,
        synth_teacher="stub:teacher",
        synth_strategy=strategy,
        synth_at="2026-04-23T20:00:00Z",
        source_section_id=source_section_id,
    )
    return PlannedSynthInstruction(
        source=SynthSourceSection(section_id=source_section_id, content="source prose"),
        strategy=strategy,  # type: ignore[arg-type]
        pair=pair,
        section=section,
    )


class StubJudge:
    name = "stub:judge"
    suggested_threshold = 0.1

    def __init__(self, scores: dict[tuple[str, str, str], PairScore]) -> None:
        self._scores = scores

    def score_pair(self, prompt: str, candidate_a: str, candidate_b: str) -> PairScore:
        return self._scores[(prompt, candidate_a, candidate_b)]


class TestFilterSynthPlan:
    def test_negative_threshold_is_rejected(self) -> None:
        raw = SynthRunPlan(additions=(_planned(),), skipped=())

        with pytest.raises(ValueError, match="threshold must be >= 0.0"):
            filter_synth_plan(raw, filter_kind="sway", judge=StubJudge({}), threshold=-0.1)

    def test_none_filter_keeps_deduped_additions(self) -> None:
        raw = SynthRunPlan(
            additions=(
                _planned(question="Q1", answer="A1"),
                _planned(source_section_id="bbbbbbbbbbbbbbbb", question="Q2", answer="A2"),
            ),
            skipped=(),
        )

        filtered = filter_synth_plan(raw, filter_kind="none")

        assert len(filtered.additions) == 2
        assert filtered.filtered_skipped == ()
        assert filtered.report.generated_count == 2
        assert filtered.report.dedup_count == 2
        assert filtered.report.accepted_count == 2

    def test_dedup_only_removes_near_duplicates(self) -> None:
        raw = SynthRunPlan(
            additions=(
                _planned(
                    question="What is DGEMM?",
                    answer="A matrix multiply routine.",
                ),
                _planned(
                    source_section_id="bbbbbbbbbbbbbbbb",
                    question="What is DGEMM?",
                    answer="A matrix multiply routine!",
                ),
            ),
            skipped=(),
        )

        filtered = filter_synth_plan(raw, filter_kind="dedup-only")

        assert len(filtered.additions) == 1
        assert len(filtered.filtered_skipped) == 1
        assert filtered.filtered_skipped[0].reason.value == "duplicate_pair"
        assert filtered.report.generated_count == 2
        assert filtered.report.dedup_count == 1
        assert filtered.report.accepted_count == 1

    def test_dedup_only_removes_near_duplicates_by_similarity(self) -> None:
        raw = SynthRunPlan(
            additions=(
                _planned(
                    question="What does DGEMM compute?",
                    answer="A dense matrix product.",
                ),
                _planned(
                    source_section_id="bbbbbbbbbbbbbbbb",
                    question="What does DGEMM compute",
                    answer="A dense matrix product.",
                ),
            ),
            skipped=(),
        )

        filtered = filter_synth_plan(raw, filter_kind="dedup-only")

        assert len(filtered.additions) == 1
        assert len(filtered.filtered_skipped) == 1
        assert filtered.filtered_skipped[0].reason.value == "duplicate_pair"

    def test_sway_filter_uses_judge_and_threshold(self) -> None:
        first = _planned(question="Q1", answer="A1")
        second = _planned(source_section_id="bbbbbbbbbbbbbbbb", question="Q2", answer="A2")
        raw = SynthRunPlan(additions=(first, second), skipped=())
        judge = StubJudge(
            {
                ("Q1", "A1", "I don't know."): PairScore(score_a=0.9, score_b=0.1),
                ("Q2", "A2", "I don't know."): PairScore(score_a=0.55, score_b=0.50),
            }
        )

        filtered = filter_synth_plan(raw, filter_kind="sway", judge=judge, threshold=0.1)

        assert len(filtered.additions) == 1
        assert filtered.additions[0].judge_score is not None
        assert len(filtered.filtered_skipped) == 1
        assert filtered.filtered_skipped[0].reason.value == "below_threshold"
        assert filtered.report.generated_count == 2
        assert filtered.report.dedup_count == 2
        assert filtered.report.judge_pass_count == 2
        assert filtered.report.accepted_count == 1

    def test_sway_filter_rejects_when_judge_does_not_prefer_generated_answer(self) -> None:
        addition = _planned(question="Q1", answer="A1")
        raw = SynthRunPlan(additions=(addition,), skipped=())
        judge = StubJudge(
            {
                ("Q1", "A1", "I don't know."): PairScore(score_a=0.0, score_b=0.4),
            }
        )

        filtered = filter_synth_plan(raw, filter_kind="sway", judge=judge)

        assert filtered.additions == ()
        assert len(filtered.filtered_skipped) == 1
        assert filtered.filtered_skipped[0].reason.value == "judge_disfavored"
        assert filtered.report.judge_pass_count == 0

    def test_sway_requires_judge(self) -> None:
        raw = SynthRunPlan(additions=(_planned(),), skipped=())
        with pytest.raises(ValueError, match="requires a judge"):
            filter_synth_plan(raw, filter_kind="sway")

    def test_render_filter_report_mentions_pipeline_counts(self) -> None:
        raw = SynthRunPlan(additions=(_planned(),), skipped=())
        judge = StubJudge(
            {
                ("What is DGEMM?", "A matrix multiply routine.", "I don't know."): PairScore(
                    score_a=0.9,
                    score_b=0.1,
                )
            }
        )
        filtered = filter_synth_plan(raw, filter_kind="sway", judge=judge)
        rendered = render_filter_report(filtered)

        assert "generated 1, dedup 1, judge passed 1, threshold 1" in rendered

    def test_render_filter_report_for_dedup_only_mentions_filtered_entries(self) -> None:
        raw = SynthRunPlan(
            additions=(
                _planned(question="What is DGEMM?", answer="A matrix multiply routine."),
                _planned(
                    source_section_id="bbbbbbbbbbbbbbbb",
                    question="What is DGEMM?",
                    answer="A matrix multiply routine!",
                ),
            ),
            skipped=(),
        )

        filtered = filter_synth_plan(raw, filter_kind="dedup-only")
        rendered = render_filter_report(filtered)

        assert "generated 2, dedup 1, accepted 1" in rendered
        assert "=== filtered ===" in rendered
        assert "duplicate_pair" in rendered
