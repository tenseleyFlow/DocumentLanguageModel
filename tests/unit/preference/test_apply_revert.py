"""Apply/revert substrate for auto-mined preference sections."""

from __future__ import annotations

from pathlib import Path

from dlm.doc.parser import parse_file, parse_text
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import serialize
from dlm.preference import (
    PreferenceSkipReason,
    apply_plan,
    build_apply_plan,
    render_apply_plan,
    revert_all_auto_mined,
)

_FRONTMATTER = """---
dlm_id: 01KPQ9X1000000000000000000
dlm_version: 14
base_model: smollm2-135m
---
"""


def _write_dlm(path: Path, body: str = "") -> None:
    path.write_text(_FRONTMATTER + body, encoding="utf-8")


def _mined_pref(
    *,
    prompt: str = "question?",
    chosen: str = "better",
    rejected: str = "worse",
    run_id: int = 7,
) -> Section:
    body = f"### Prompt\n{prompt}\n### Chosen\n{chosen}\n### Rejected\n{rejected}"
    return Section(
        type=SectionType.PREFERENCE,
        content=body,
        auto_mined=True,
        judge_name="sway:preference_judge",
        judge_score_chosen=0.9,
        judge_score_rejected=0.1,
        mined_at="2026-04-23T20:00:00Z",
        mined_run_id=run_id,
    )


def _authored_pref() -> Section:
    return Section(
        type=SectionType.PREFERENCE,
        content="### Prompt\nmanual\n### Chosen\nyes\n### Rejected\nno",
    )


def _instruction() -> Section:
    return Section(
        type=SectionType.INSTRUCTION,
        content="### Q\nhi?\n### A\nhello.",
    )


class TestBuildApplyPlan:
    def test_accepts_new_auto_mined_preference(self) -> None:
        parsed = parse_text(_FRONTMATTER + "prose body\n")
        plan = build_apply_plan(parsed, [_mined_pref()])

        assert len(plan.additions) == 1
        assert plan.skipped == ()
        assert plan.additions[0].section.auto_mined is True

    def test_dedupes_within_input(self) -> None:
        parsed = parse_text(_FRONTMATTER + "prose body\n")
        section = _mined_pref()
        plan = build_apply_plan(parsed, [section, section])

        assert len(plan.additions) == 1
        assert len(plan.skipped) == 1
        assert plan.skipped[0].reason is PreferenceSkipReason.ALREADY_PRESENT

    def test_skips_non_preference_and_hand_authored(self) -> None:
        parsed = parse_text(_FRONTMATTER + "prose body\n")
        plan = build_apply_plan(parsed, [_instruction(), _authored_pref()])

        assert plan.additions == ()
        assert [skip.reason for skip in plan.skipped] == [
            PreferenceSkipReason.NOT_PREFERENCE,
            PreferenceSkipReason.NOT_AUTO_MINED,
        ]

    def test_render_plan_mentions_adds_and_skips(self) -> None:
        parsed = parse_text(_FRONTMATTER + "prose body\n")
        plan = build_apply_plan(parsed, [_mined_pref(), _authored_pref()])
        rendered = render_apply_plan(plan)

        assert "1 add, 1 skip" in rendered
        assert "::preference::" in rendered
        assert "not_auto_mined" in rendered


class TestApplyPlan:
    def test_writes_additions_and_preserves_body(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        _write_dlm(target, "## hello\n\nkeep me\n")

        parsed = parse_file(target)
        plan = build_apply_plan(parsed, [_mined_pref()])
        summary = apply_plan(parsed, plan, target=target)

        assert summary.added == 1
        assert summary.skipped == 0
        assert len(summary.added_section_ids) == 1

        reloaded = parse_file(target)
        assert any(section.auto_mined for section in reloaded.sections)
        assert any("keep me" in section.content for section in reloaded.sections)

    def test_existing_document_section_is_skipped(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        existing = _mined_pref()
        parsed = parse_text(_FRONTMATTER, path=target)
        plan = build_apply_plan(parsed, [existing])
        apply_plan(parsed, plan, target=target)

        reloaded = parse_file(target)
        second_plan = build_apply_plan(reloaded, [existing])
        assert second_plan.additions == ()
        assert len(second_plan.skipped) == 1
        assert second_plan.skipped[0].reason is PreferenceSkipReason.ALREADY_PRESENT


class TestRevertAutoMined:
    def test_strips_only_auto_mined_preferences(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        _write_dlm(target, "## hello\n\nkeep me\n")
        parsed = parse_file(target)
        plan = build_apply_plan(parsed, [_mined_pref()])
        apply_plan(parsed, plan, target=target)

        reloaded = parse_file(target)
        updated = type(reloaded)(
            frontmatter=reloaded.frontmatter,
            sections=reloaded.sections + (_authored_pref(), _instruction()),
            source_path=reloaded.source_path,
        )
        target.write_text(serialize(updated), encoding="utf-8")

        parsed_with_all = parse_file(target)
        summary = revert_all_auto_mined(parsed_with_all, target=target)

        assert summary.added == 0
        assert len(summary.added_section_ids) == 1

        final = parse_file(target)
        assert not any(section.auto_mined for section in final.sections)
        assert any(section.type is SectionType.PREFERENCE for section in final.sections)
        assert any(section.type is SectionType.INSTRUCTION for section in final.sections)
        assert any("keep me" in section.content for section in final.sections)

    def test_revert_noop_when_no_auto_mined(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        _write_dlm(
            target, "::preference::\n### Prompt\nmanual\n### Chosen\nyes\n### Rejected\nno\n"
        )
        parsed = parse_file(target)
        summary = revert_all_auto_mined(parsed, target=target)

        assert summary.added == 0
        assert summary.added_section_ids == ()
        reloaded = parse_file(target)
        assert len(reloaded.sections) == len(parsed.sections)
