"""Dry-run synth planning for Sprint 43."""

from __future__ import annotations

from collections import deque

import pytest

import dlm.synth.run as run_mod
from dlm.doc.parser import ParsedDlm, parse_text
from dlm.synth import SynthPromptTemplate, build_synth_plan, render_synth_plan

_FRONTMATTER = """---
dlm_id: 01KPQ9X1000000000000000000
dlm_version: 15
base_model: smollm2-135m
---
"""


class StubTeacher:
    def __init__(self, outputs: list[str]) -> None:
        self.name = "stub:teacher"
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self._outputs = deque(outputs)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        self.calls.append(
            (
                system_prompt,
                user_prompt,
                {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                },
            )
        )
        return self._outputs.popleft()


def _parsed(body: str) -> ParsedDlm:
    return parse_text(_FRONTMATTER + body)


class TestBuildSynthPlan:
    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("per_section", 0, "per_section must be >= 1"),
            ("max_pairs", 0, "max_pairs must be >= 1"),
            ("max_new_tokens", 0, "max_new_tokens must be >= 1"),
        ],
    )
    def test_rejects_invalid_limits(
        self,
        field: str,
        value: int,
        message: str,
    ) -> None:
        parsed = _parsed("One prose block.\n")

        if field == "per_section":
            with pytest.raises(ValueError, match=message):
                build_synth_plan(parsed, StubTeacher([]), per_section=value)
        elif field == "max_pairs":
            with pytest.raises(ValueError, match=message):
                build_synth_plan(parsed, StubTeacher([]), max_pairs=value)
        else:
            with pytest.raises(ValueError, match=message):
                build_synth_plan(parsed, StubTeacher([]), max_new_tokens=value)

    def test_materializes_auto_synth_instruction_sections(self) -> None:
        parsed = _parsed("A short prose section about matrix multiplication.\n")
        teacher = StubTeacher(
            [
                '[{"question":"What is matrix multiplication?","answer":"An operation over matrices."},'
                '{"question":"Why is it useful?","answer":"It composes linear transforms."}]'
            ]
        )

        plan = build_synth_plan(
            parsed,
            teacher,
            per_section=2,
            strategy="extraction",
            synth_at="2026-04-23T20:00:00Z",
        )

        assert len(plan.additions) == 2
        assert plan.skipped == ()
        first = plan.additions[0]
        assert first.source.section_id == parsed.sections[0].section_id
        assert first.section.auto_synth is True
        assert first.section.synth_teacher == "stub:teacher"
        assert first.section.synth_strategy == "extraction"
        assert first.section.synth_at == "2026-04-23T20:00:00Z"
        assert first.section.source_section_id == parsed.sections[0].section_id
        assert first.section.content == (
            "### Q\nWhat is matrix multiplication?\n### A\nAn operation over matrices."
        )
        assert len(teacher.calls) == 1

    def test_both_strategy_splits_calls_across_templates(self) -> None:
        parsed = _parsed("One prose block.\n")
        teacher = StubTeacher(
            [
                '[{"question":"Extraction 1","answer":"A1"},{"question":"Extraction 2","answer":"A2"}]',
                '[{"question":"Expansion 1","answer":"B1"}]',
            ]
        )

        plan = build_synth_plan(
            parsed,
            teacher,
            per_section=3,
            strategy="both",
            synth_at="2026-04-23T20:00:00Z",
        )

        assert [add.strategy for add in plan.additions] == [
            "extraction",
            "extraction",
            "expansion",
        ]
        assert len(teacher.calls) == 2
        assert "explicitly supported by the prose" in teacher.calls[0][1]
        assert "expand on the material" in teacher.calls[1][1]

    def test_invalid_teacher_output_is_skipped(self) -> None:
        parsed = _parsed("One prose block.\n")
        teacher = StubTeacher(["not json"])

        plan = build_synth_plan(parsed, teacher, per_section=1, strategy="extraction")

        assert plan.additions == ()
        assert len(plan.skipped) == 1
        assert plan.skipped[0].reason.value == "invalid_output"

    def test_json_output_wrapped_in_code_fence_is_accepted(self) -> None:
        parsed = _parsed("One prose block.\n")
        teacher = StubTeacher(
            ['```json\n[{"question":"What is DGEMM?","answer":"Matrix multiply."}]\n```']
        )

        plan = build_synth_plan(parsed, teacher, per_section=1, strategy="extraction")

        assert len(plan.additions) == 1
        assert plan.additions[0].pair.question == "What is DGEMM?"
        assert plan.additions[0].pair.answer == "Matrix multiply."

    def test_existing_instruction_duplicate_is_skipped(self) -> None:
        parsed = _parsed(
            "Prose section.\n\n"
            "::instruction::\n"
            "### Q\nWhat is a quine?\n"
            "### A\nA program that prints itself.\n"
        )
        teacher = StubTeacher(
            ['[{"question":"What is a quine?","answer":"A program that prints itself."}]']
        )

        plan = build_synth_plan(parsed, teacher, per_section=1, strategy="extraction")

        assert plan.additions == ()
        assert len(plan.skipped) == 1
        assert plan.skipped[0].reason.value == "already_present"

    def test_no_prose_sections_returns_document_level_skip(self) -> None:
        parsed = _parsed("::instruction::\n### Q\nhi?\n### A\nhello.\n")
        teacher = StubTeacher([])

        plan = build_synth_plan(parsed, teacher, per_section=1, strategy="extraction")

        assert plan.additions == ()
        assert len(plan.skipped) == 1
        assert plan.skipped[0].reason.value == "no_prose"

    def test_numbered_list_parser_supported_via_template_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parsed = _parsed("Prose block.\n")
        teacher = StubTeacher(["1. Q: What is DGEMM?\nA: Matrix multiply.\n"])

        template = SynthPromptTemplate(
            system_prompt="Return a numbered list.",
            user_template="Use the prose below to generate {{ n }} pairs.\n\n{{ prose }}",
            output_parser="numbered_list",
        )
        monkeypatch.setattr(run_mod, "get_prompt_template", lambda strategy: template)

        plan = build_synth_plan(parsed, teacher, per_section=1, strategy="extraction")

        assert len(plan.additions) == 1
        assert plan.additions[0].pair.question == "What is DGEMM?"
        assert plan.additions[0].pair.answer == "Matrix multiply."

    def test_max_pairs_caps_output_across_sections(self) -> None:
        parsed = _parsed("First prose.\n\nSecond prose.\n")
        teacher = StubTeacher(
            [
                '[{"question":"Q1","answer":"A1"}]',
                '[{"question":"Q2","answer":"A2"}]',
            ]
        )

        plan = build_synth_plan(parsed, teacher, per_section=1, strategy="extraction", max_pairs=1)

        assert len(plan.additions) == 1
        assert plan.additions[0].pair.question == "Q1"
        assert len(teacher.calls) == 1

    def test_max_pairs_returns_before_generating_from_later_sources(self) -> None:
        parsed = _parsed(
            "First prose block.\n\n"
            "::instruction::\n"
            "### Q\nmanual?\n"
            "### A\nyes.\n\n"
            "Second prose block.\n"
        )
        teacher = StubTeacher(
            [
                '[{"question":"Q1","answer":"A1"}]',
                '[{"question":"Q2","answer":"A2"}]',
            ]
        )

        plan = build_synth_plan(parsed, teacher, per_section=1, strategy="extraction", max_pairs=1)

        assert len(plan.additions) == 1
        assert len(teacher.calls) == 1

    def test_both_strategy_skips_zero_count_branch(self) -> None:
        parsed = _parsed("One prose block.\n")
        teacher = StubTeacher(['[{"question":"Q1","answer":"A1"}]'])

        plan = build_synth_plan(parsed, teacher, per_section=1, strategy="both")

        assert len(plan.additions) == 1
        assert [add.strategy for add in plan.additions] == ["extraction"]
        assert len(teacher.calls) == 1

    def test_expansion_strategy_uses_expansion_template(self) -> None:
        parsed = _parsed("One prose block.\n")
        teacher = StubTeacher(['[{"question":"Q1","answer":"A1"}]'])

        plan = build_synth_plan(parsed, teacher, per_section=1, strategy="expansion")

        assert len(plan.additions) == 1
        assert [add.strategy for add in plan.additions] == ["expansion"]
        assert "expand on the material" in teacher.calls[0][1]


def test_render_synth_plan_mentions_adds_and_skips() -> None:
    parsed = _parsed("One prose block.\n")
    teacher = StubTeacher(["not json"])

    plan = build_synth_plan(parsed, teacher, per_section=1, strategy="extraction")
    rendered = render_synth_plan(plan)

    assert "synth plan: 0 add, 1 skip" in rendered
    assert "invalid_output" in rendered


def test_render_synth_plan_mentions_additions_and_truncates_long_lines() -> None:
    parsed = _parsed("One prose block.\n")
    long_question = "Q" * 90
    long_answer = "A" * 90
    teacher = StubTeacher([f'[{{"question":"{long_question}","answer":"{long_answer}"}}]'])

    plan = build_synth_plan(parsed, teacher, per_section=1, strategy="extraction")
    rendered = render_synth_plan(plan)

    assert "synth plan: 1 add, 0 skip" in rendered
    assert "+ ::instruction::" in rendered
    assert "q: " in rendered
    assert "a: " in rendered
    assert "…" in rendered


def test_first_line_returns_short_text_unchanged() -> None:
    assert run_mod._first_line("short line") == "short line"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("[]", "teacher output produced no instruction pairs"),
        ("{}", "teacher output must be a JSON list"),
        ("[1]", "teacher output item 0 must be an object"),
        ('[{"question":1,"answer":"ok"}]', "must contain string question/answer keys"),
        ('[{"question":" ","answer":"ok"}]', "has an empty question or answer"),
    ],
)
def test_parse_generated_pairs_rejects_bad_json_list_payloads(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_mod._parse_generated_pairs(raw, parser="json_list")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("Question: hi\nA: ok", "must use lines like `1. Q: ...`"),
        ("1. Q: hi", "missing an answer line"),
        ("1. Q: hi\nB: ok", "answers must use `A:` or `Answer:`"),
        ("1. Q:   \nA: ok", "contains an empty question or answer"),
    ],
)
def test_parse_generated_pairs_rejects_bad_numbered_list_payloads(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_mod._parse_generated_pairs(raw, parser="numbered_list")
