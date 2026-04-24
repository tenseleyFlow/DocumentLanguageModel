"""Tests for synth apply/revert and pending-plan helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlm.doc.parser import ParsedDlm, parse_file, parse_text
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import serialize
from dlm.store.paths import for_dlm
from dlm.synth.apply import (
    SynthApplySkipReason,
    apply_plan,
    build_apply_plan,
    render_apply_plan,
    revert_all_auto_synth,
)
from dlm.synth.pending import (
    PendingSynthPlanError,
    _optional_float,
    _optional_int,
    _optional_str,
    _section_from_payload,
    clear_pending_plan,
    load_pending_plan,
    pending_plan_path,
    save_pending_plan,
)

_DLM_ID = "01KPQ9X1000000000000000000"
_FRONTMATTER = f"---\ndlm_id: {_DLM_ID}\ndlm_version: 15\nbase_model: smollm2-135m\n---\n"


def _write_dlm(path: Path, body: str = "") -> None:
    path.write_text(_FRONTMATTER + body, encoding="utf-8")


def _auto_synth_instruction(
    *,
    question: str = "What does DGEMM do?",
    answer: str = "It multiplies dense matrices.",
    teacher: str = "self",
    strategy: str = "extraction",
    source_section_id: str = "0123456789abcdef",
) -> Section:
    return Section(
        type=SectionType.INSTRUCTION,
        content=f"### Q\n{question}\n### A\n{answer}",
        start_line=12,
        adapter="tone",
        tags={"topic": "blas"},
        auto_synth=True,
        synth_teacher=teacher,
        synth_strategy=strategy,
        synth_at="2026-04-24T20:00:00Z",
        source_section_id=source_section_id,
    )


def _authored_instruction() -> Section:
    return Section(
        type=SectionType.INSTRUCTION,
        content="### Q\nWhat is BLAS?\n### A\nA linear algebra interface.",
    )


def _preference() -> Section:
    return Section(
        type=SectionType.PREFERENCE,
        content="### Prompt\nmanual\n### Chosen\nyes\n### Rejected\nno",
    )


def _image() -> Section:
    return Section(
        type=SectionType.IMAGE,
        content="A DGEMM block diagram.",
        media_path="diagram.png",
        media_alt="DGEMM diagram",
        media_blob_sha="ab" * 32,
    )


class TestBuildApplyPlan:
    def test_accepts_new_auto_synth_instruction(self) -> None:
        parsed = parse_text(_FRONTMATTER + "prose body\n")
        plan = build_apply_plan(parsed, [_auto_synth_instruction()])

        assert len(plan.additions) == 1
        assert plan.skipped == ()
        assert plan.additions[0].section.auto_synth is True

    def test_dedupes_within_input(self) -> None:
        parsed = parse_text(_FRONTMATTER + "prose body\n")
        section = _auto_synth_instruction()
        plan = build_apply_plan(parsed, [section, section])

        assert len(plan.additions) == 1
        assert len(plan.skipped) == 1
        assert plan.skipped[0].reason is SynthApplySkipReason.ALREADY_PRESENT

    def test_skips_non_instruction_and_hand_authored(self) -> None:
        parsed = parse_text(_FRONTMATTER + "prose body\n")
        plan = build_apply_plan(parsed, [_preference(), _authored_instruction()])

        assert plan.additions == ()
        assert [skip.reason for skip in plan.skipped] == [
            SynthApplySkipReason.NOT_INSTRUCTION,
            SynthApplySkipReason.NOT_AUTO_SYNTH,
        ]

    def test_render_plan_mentions_adds_and_skips(self) -> None:
        parsed = parse_text(_FRONTMATTER + "prose body\n")
        plan = build_apply_plan(parsed, [_auto_synth_instruction(), _authored_instruction()])
        rendered = render_apply_plan(plan)

        assert "1 add, 1 skip" in rendered
        assert "::instruction::" in rendered
        assert "teacher=self" in rendered
        assert "strategy=extraction" in rendered
        assert "source=0123456789abcdef" in rendered
        assert "not_auto_synth" in rendered


class TestApplyPlan:
    def test_writes_additions_and_preserves_body(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        _write_dlm(target, "## hello\n\nkeep me\n")

        parsed = parse_file(target)
        plan = build_apply_plan(parsed, [_auto_synth_instruction()])
        summary = apply_plan(parsed, plan, target=target)

        assert summary.added == 1
        assert summary.skipped == 0
        assert len(summary.added_section_ids) == 1

        reloaded = parse_file(target)
        assert any(section.auto_synth for section in reloaded.sections)
        assert any("keep me" in section.content for section in reloaded.sections)

    def test_existing_document_section_is_skipped(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        existing = _auto_synth_instruction()
        parsed = parse_text(_FRONTMATTER, path=target)
        plan = build_apply_plan(parsed, [existing])
        apply_plan(parsed, plan, target=target)

        reloaded = parse_file(target)
        second_plan = build_apply_plan(reloaded, [existing])
        assert second_plan.additions == ()
        assert len(second_plan.skipped) == 1
        assert second_plan.skipped[0].reason is SynthApplySkipReason.ALREADY_PRESENT


class TestRevertAutoSynth:
    def test_strips_only_auto_synth_instructions(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        _write_dlm(target, "## hello\n\nkeep me\n")
        parsed = parse_file(target)
        plan = build_apply_plan(parsed, [_auto_synth_instruction()])
        apply_plan(parsed, plan, target=target)

        reloaded = parse_file(target)
        updated = ParsedDlm(
            frontmatter=reloaded.frontmatter,
            sections=reloaded.sections + (_authored_instruction(), _preference()),
            source_path=reloaded.source_path,
        )
        target.write_text(serialize(updated), encoding="utf-8")

        parsed_with_all = parse_file(target)
        summary = revert_all_auto_synth(parsed_with_all, target=target)

        assert summary.added == 0
        assert len(summary.added_section_ids) == 1

        final = parse_file(target)
        assert not any(section.auto_synth for section in final.sections)
        assert any(section.type is SectionType.PREFERENCE for section in final.sections)
        assert any(
            section.type is SectionType.INSTRUCTION and not section.auto_synth
            for section in final.sections
        )
        assert any("keep me" in section.content for section in final.sections)

    def test_revert_noop_when_no_auto_synth(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        _write_dlm(target, "::instruction::\n### Q\nmanual?\n### A\nyes\n")
        parsed = parse_file(target)
        summary = revert_all_auto_synth(parsed, target=target)

        assert summary.added == 0
        assert summary.added_section_ids == ()
        reloaded = parse_file(target)
        assert len(reloaded.sections) == len(parsed.sections)


class TestPendingPlan:
    def test_pending_path_round_trip_and_clear(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        source_path = tmp_path / "doc.dlm"
        _write_dlm(source_path)
        store = for_dlm(_DLM_ID, home=home)

        path = pending_plan_path(store)
        assert path == home / "store" / _DLM_ID / "synth" / "pending.json"

        saved = save_pending_plan(
            store,
            source_path=source_path,
            sections=[_auto_synth_instruction(), _image()],
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
        loaded = load_pending_plan(store)

        assert saved.source_path == source_path.resolve()
        assert saved.created_at.endswith("Z")
        assert raw["schema_version"] == 1
        assert raw["source_path"] == str(source_path.resolve())
        assert loaded == saved
        assert clear_pending_plan(store) is True
        assert clear_pending_plan(store) is False
        assert load_pending_plan(store) is None

    def test_load_returns_none_when_plan_absent(self, tmp_path: Path) -> None:
        store = for_dlm(_DLM_ID, home=tmp_path / "home")

        assert load_pending_plan(store) is None

    def test_load_rejects_unreadable_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = for_dlm(_DLM_ID, home=tmp_path / "home")
        path = pending_plan_path(store)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

        def _raise(_self: Path, *, encoding: str) -> str:
            _ = encoding
            raise OSError("boom")

        monkeypatch.setattr(Path, "read_text", _raise)
        with pytest.raises(PendingSynthPlanError, match="could not read staged synth plan"):
            load_pending_plan(store)

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            (["not", "an", "object"], "must be a JSON object"),
            ({"schema_version": 2}, "unsupported staged synth plan schema_version=2"),
            (
                {"schema_version": 1, "created_at": "2026-04-24T20:00:00Z", "sections": []},
                "missing source_path",
            ),
            (
                {"schema_version": 1, "source_path": "/tmp/doc.dlm", "sections": []},
                "missing created_at",
            ),
            (
                {
                    "schema_version": 1,
                    "source_path": "/tmp/doc.dlm",
                    "created_at": "2026-04-24T20:00:00Z",
                },
                "missing sections",
            ),
            (
                {
                    "schema_version": 1,
                    "source_path": "/tmp/doc.dlm",
                    "created_at": "2026-04-24T20:00:00Z",
                    "sections": [{"content": "oops"}],
                },
                "invalid section payload at index 0",
            ),
        ],
    )
    def test_load_rejects_invalid_payloads(
        self, tmp_path: Path, payload: object, message: str
    ) -> None:
        store = for_dlm(_DLM_ID, home=tmp_path / "home")
        path = pending_plan_path(store)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(PendingSynthPlanError, match=message):
            load_pending_plan(store)

    def test_load_rejects_invalid_json(self, tmp_path: Path) -> None:
        store = for_dlm(_DLM_ID, home=tmp_path / "home")
        path = pending_plan_path(store)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(PendingSynthPlanError, match="staged synth plan is not valid JSON"):
            load_pending_plan(store)


class TestPendingPayloadHelpers:
    def test_section_from_payload_validates_tags_and_optional_types(self) -> None:
        with pytest.raises(TypeError, match="expected object, got list"):
            _section_from_payload([])

        with pytest.raises(TypeError, match="tags must be an object"):
            _section_from_payload({"type": "instruction", "content": "x", "tags": []})

        with pytest.raises(TypeError, match="tags keys and values must be strings"):
            _section_from_payload({"type": "instruction", "content": "x", "tags": {"topic": 1}})

        with pytest.raises(TypeError, match="expected float or null"):
            _section_from_payload(
                {"type": "instruction", "content": "x", "judge_score_chosen": True}
            )

        with pytest.raises(TypeError, match="expected int or null"):
            _section_from_payload({"type": "instruction", "content": "x", "mined_run_id": True})

    def test_optional_helpers_accept_none_and_reject_wrong_types(self) -> None:
        assert _optional_str(None) is None
        assert _optional_str("ok") == "ok"
        assert _optional_float(None) is None
        assert _optional_float(1) == 1.0
        assert _optional_int(None) is None
        assert _optional_int(7) == 7

        with pytest.raises(TypeError, match="expected string or null"):
            _optional_str(7)

        with pytest.raises(TypeError, match="expected float or null"):
            _optional_float(True)

        with pytest.raises(TypeError, match="expected int or null"):
            _optional_int(True)
