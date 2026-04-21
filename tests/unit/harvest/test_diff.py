"""Unit tests for `dlm.harvest.diff` (Sprint 33.3)."""

from __future__ import annotations

from dlm.doc.parser import parse_text
from dlm.harvest import (
    HarvestCandidate,
    SkipReason,
    build_plan,
    render_plan,
)

_FRONTMATTER = """---
dlm_id: 01KPQ9X1000000000000000000
dlm_version: 7
base_model: smollm2-135m
---
"""


def _parsed(body: str = ""):
    return parse_text(_FRONTMATTER + body)


def _candidate(
    prompt: str = "what is a quine?",
    reference: str = "a program that prints itself.",
    probe_name: str = "quine_q",
    confidence: float = 0.9,
) -> HarvestCandidate:
    return HarvestCandidate(
        prompt=prompt,
        reference=reference,
        confidence=confidence,
        probe_name=probe_name,
        probe_kind="section_internalization",
        source_adapter_version="run_7",
    )


class TestBuildPlan:
    def test_empty_candidates_yields_empty_plan(self) -> None:
        plan = build_plan(_parsed(), [])
        assert plan.additions == ()
        assert plan.skipped == ()

    def test_single_candidate_becomes_instruction_section(self) -> None:
        plan = build_plan(_parsed(), [_candidate()])
        assert len(plan.additions) == 1
        add = plan.additions[0]
        section = add.section
        assert section.type.value == "instruction"
        assert section.auto_harvest is True
        assert section.harvest_source == "auto-harvest/quine_q"
        assert "!probe" in section.content
        assert "### Q\nwhat is a quine?" in section.content
        assert "### A\na program that prints itself." in section.content

    def test_custom_tag_prefixes_harvest_source(self) -> None:
        plan = build_plan(_parsed(), [_candidate()], tag="sway-ci")
        assert plan.additions[0].section.harvest_source == "sway-ci/quine_q"

    def test_dedup_against_existing_section(self) -> None:
        # First build a plan to get the section the harvest would emit;
        # include it in the doc; rebuild — dedup should skip it.
        first = build_plan(_parsed(), [_candidate()])
        existing_section = first.additions[0].section
        body = (
            "::instruction::\n"
            '<!-- dlm-auto-harvest: source="auto-harvest/quine_q" -->\n'
            + existing_section.content
            + "\n"
        )
        plan = build_plan(_parsed(body), [_candidate()])
        assert plan.additions == ()
        assert len(plan.skipped) == 1
        assert plan.skipped[0].reason is SkipReason.ALREADY_PRESENT

    def test_dedup_within_plan(self) -> None:
        plan = build_plan(_parsed(), [_candidate(), _candidate()])
        assert len(plan.additions) == 1
        assert len(plan.skipped) == 1
        assert plan.skipped[0].reason is SkipReason.ALREADY_PRESENT


class TestRenderPlan:
    def test_renders_addition_summary(self) -> None:
        plan = build_plan(_parsed(), [_candidate()])
        text = render_plan(plan)
        assert "harvest plan: 1 add, 0 skip" in text
        assert "=== additions ===" in text
        assert "probe=quine_q" in text
        assert "what is a quine?" in text

    def test_renders_skipped(self) -> None:
        plan = build_plan(_parsed(), [_candidate(), _candidate()])
        text = render_plan(plan)
        assert "=== skipped ===" in text
        assert "already_present" in text

    def test_truncates_long_prompts(self) -> None:
        long_prompt = "x" * 200
        plan = build_plan(_parsed(), [_candidate(prompt=long_prompt)])
        text = render_plan(plan)
        # rendered to ~80 chars with an ellipsis
        assert "…" in text
