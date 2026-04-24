"""Unit tests for Sprint 42's pure preference-mining backend."""

from __future__ import annotations

from collections import deque

from dlm.doc.parser import parse_text
from dlm.preference import (
    PreferenceMineSkipReason,
    build_mine_plan,
    render_mine_plan,
)
from dlm.preference.judge import PairScore

_FRONTMATTER = """---
dlm_id: 01KPQ9X1000000000000000000
dlm_version: 14
base_model: smollm2-135m
---
"""


class StubBackend:
    def __init__(self, responses: dict[str, list[str]]) -> None:
        self._responses = {prompt: deque(items) for prompt, items in responses.items()}

    def generate(self, prompt: str, **_gen_kwargs: object) -> str:
        return self._responses[prompt].popleft()


class StubJudge:
    name = "stub:judge"
    suggested_threshold = 0.10

    def __init__(self, scores: dict[tuple[str, str, str], PairScore]) -> None:
        self._scores = scores

    def score_pair(self, prompt: str, candidate_a: str, candidate_b: str) -> PairScore:
        return self._scores[(prompt, candidate_a, candidate_b)]


def _parsed(body: str):
    return parse_text(_FRONTMATTER + body)


class TestBuildMinePlan:
    def test_materializes_auto_mined_preference_section(self) -> None:
        parsed = _parsed("::instruction::\n### Q\nquestion?\n### A\nreference\n")
        backend = StubBackend({"question?": ["bad answer", "good answer"]})
        judge = StubJudge(
            {
                ("question?", "bad answer", "good answer"): PairScore(
                    score_a=0.1,
                    score_b=0.9,
                    reasoning="good answer wins clearly",
                )
            }
        )

        plan = build_mine_plan(
            parsed,
            backend,
            judge,
            mined_run_id=7,
            samples=2,
            mined_at="2026-04-23T20:00:00Z",
        )

        assert len(plan.additions) == 1
        assert plan.skipped == ()
        addition = plan.additions[0]
        assert addition.chosen == "good answer"
        assert addition.rejected == "bad answer"
        assert addition.source.prompt == "question?"
        assert addition.section.auto_mined is True
        assert addition.section.judge_name == "stub:judge"
        assert addition.section.judge_score_chosen == 0.9
        assert addition.section.judge_score_rejected == 0.1
        assert addition.section.mined_at == "2026-04-23T20:00:00Z"
        assert addition.section.mined_run_id == 7
        assert addition.section.content == (
            "### Prompt\nquestion?\n### Chosen\ngood answer\n### Rejected\nbad answer"
        )

    def test_probe_markers_are_normalized_and_duplicate_prompts_skip(self) -> None:
        parsed = _parsed(
            "::instruction::\n"
            "### Q !probe\n"
            "What is DGEMM?\n"
            "### A\n"
            "A matrix multiply.\n\n"
            "::instruction::\n"
            "### Q\n"
            "What is DGEMM?\n"
            "### A\n"
            "Still a matrix multiply.\n"
        )
        backend = StubBackend({"What is DGEMM?": ["weak", "strong"]})
        judge = StubJudge(
            {
                ("What is DGEMM?", "weak", "strong"): PairScore(
                    score_a=0.2,
                    score_b=0.8,
                )
            }
        )

        plan = build_mine_plan(parsed, backend, judge, mined_run_id=3, samples=2)

        assert len(plan.additions) == 1
        assert len(plan.skipped) == 1
        assert plan.skipped[0].reason is PreferenceMineSkipReason.DUPLICATE_PROMPT
        assert plan.additions[0].source.prompt == "What is DGEMM?"

    def test_existing_preference_is_not_remined(self) -> None:
        parsed = _parsed(
            "::instruction::\n"
            "### Q\n"
            "question?\n"
            "### A\n"
            "reference\n\n"
            "::preference::\n"
            "### Prompt\n"
            "question?\n"
            "### Chosen\n"
            "good answer\n"
            "### Rejected\n"
            "bad answer\n"
        )
        backend = StubBackend({"question?": ["bad answer", "good answer"]})
        judge = StubJudge(
            {
                ("question?", "bad answer", "good answer"): PairScore(
                    score_a=0.2,
                    score_b=0.9,
                )
            }
        )

        plan = build_mine_plan(parsed, backend, judge, mined_run_id=5, samples=2)

        assert plan.additions == ()
        assert len(plan.skipped) == 1
        assert plan.skipped[0].reason is PreferenceMineSkipReason.ALREADY_PRESENT

    def test_near_identical_candidates_are_rejected(self) -> None:
        parsed = _parsed("::instruction::\n### Q\nquestion?\n### A\nreference\n")
        backend = StubBackend(
            {
                "question?": [
                    "This answer explains the algorithm step by step in a calm, direct tone.",
                    "This answer explains the algorithm step by step in a calm, direct tone!",
                ]
            }
        )
        judge = StubJudge({})

        plan = build_mine_plan(parsed, backend, judge, mined_run_id=9, samples=2)

        assert plan.additions == ()
        assert len(plan.skipped) == 1
        assert plan.skipped[0].reason is PreferenceMineSkipReason.TOO_SIMILAR

    def test_below_threshold_pairs_are_skipped_and_rendered(self) -> None:
        parsed = _parsed("::instruction::\n### Q\nquestion?\n### A\nreference\n")
        backend = StubBackend({"question?": ["candidate one", "candidate two"]})
        judge = StubJudge(
            {
                ("question?", "candidate one", "candidate two"): PairScore(
                    score_a=0.52,
                    score_b=0.48,
                )
            }
        )

        plan = build_mine_plan(parsed, backend, judge, mined_run_id=11, samples=2)
        rendered = render_mine_plan(plan)

        assert plan.additions == ()
        assert len(plan.skipped) == 1
        assert plan.skipped[0].reason is PreferenceMineSkipReason.BELOW_THRESHOLD
        assert "preference mine plan: 0 add, 1 skip" in rendered
        assert "below_threshold" in rendered

    def test_malformed_instruction_section_is_reported(self) -> None:
        parsed = _parsed("::instruction::\n### Q\nunterminated question\n")
        backend = StubBackend({})
        judge = StubJudge({})

        plan = build_mine_plan(parsed, backend, judge, mined_run_id=1, samples=2)

        assert plan.additions == ()
        assert len(plan.skipped) == 1
        assert plan.skipped[0].reason is PreferenceMineSkipReason.MALFORMED_INSTRUCTION
