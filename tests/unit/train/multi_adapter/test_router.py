"""Routing: section → adapter assignment."""

from __future__ import annotations

from textwrap import dedent

import pytest

from dlm.doc.parser import parse_text
from dlm.doc.sections import SectionType
from dlm.train.multi_adapter.router import (
    UnknownAdapterError,
    build_plan,
    declared_adapter_names,
    sections_for,
)


def _doc(body: str, *, multi_adapter: bool) -> str:
    if multi_adapter:
        fm = dedent(
            """\
            ---
            dlm_id: 01HZ4X7TGZM3J1A2B3C4D5E6F7
            base_model: smollm2-135m
            training:
              adapters:
                knowledge: {}
                tone: {lora_r: 4}
            ---
            """
        )
    else:
        fm = dedent(
            """\
            ---
            dlm_id: 01HZ4X7TGZM3J1A2B3C4D5E6F7
            base_model: smollm2-135m
            ---
            """
        )
    return fm + body


class TestDeclaredAdapterNames:
    def test_single_adapter_doc_returns_default(self) -> None:
        parsed = parse_text(_doc("hello prose\n", multi_adapter=False))
        assert declared_adapter_names(parsed) == ["default"]

    def test_multi_adapter_preserves_declaration_order(self) -> None:
        parsed = parse_text(_doc("hello prose\n", multi_adapter=True))
        assert declared_adapter_names(parsed) == ["knowledge", "tone"]


class TestProseFansOut:
    def test_prose_without_suffix_goes_to_all_adapters(self) -> None:
        parsed = parse_text(_doc("# Domain\n\nShared prose.\n", multi_adapter=True))
        plan = build_plan(parsed)
        # Prose appears under both adapters.
        prose_in_knowledge = [
            s for s in plan.by_adapter["knowledge"] if s.type is SectionType.PROSE
        ]
        prose_in_tone = [s for s in plan.by_adapter["tone"] if s.type is SectionType.PROSE]
        assert len(prose_in_knowledge) == 1
        assert len(prose_in_tone) == 1

    def test_prose_with_name_only_goes_to_that_adapter(self) -> None:
        body = "::prose#knowledge::\nOnly knowledge cares.\n"
        parsed = parse_text(_doc(body, multi_adapter=True))
        plan = build_plan(parsed)
        assert any(s.adapter == "knowledge" for s in plan.by_adapter["knowledge"])
        assert all(s.adapter != "knowledge" for s in plan.by_adapter["tone"])


class TestInstructionRouting:
    def test_untagged_instruction_goes_to_first_adapter(self) -> None:
        body = "::instruction::\n### Q\nhi\n### A\nbye\n"
        parsed = parse_text(_doc(body, multi_adapter=True))
        plan = build_plan(parsed)
        # First-declared is "knowledge".
        assert any(s.type is SectionType.INSTRUCTION for s in plan.by_adapter["knowledge"])
        assert not any(s.type is SectionType.INSTRUCTION for s in plan.by_adapter["tone"])

    def test_tagged_instruction_goes_to_named_adapter(self) -> None:
        body = "::instruction#tone::\n### Q\nhi\n### A\nbye\n"
        parsed = parse_text(_doc(body, multi_adapter=True))
        plan = build_plan(parsed)
        assert not any(s.type is SectionType.INSTRUCTION for s in plan.by_adapter["knowledge"])
        assert any(s.type is SectionType.INSTRUCTION for s in plan.by_adapter["tone"])


class TestPreferenceRouting:
    def test_tagged_preference_goes_to_named_adapter(self) -> None:
        body = "::preference#tone::\n### Prompt\nq\n### Chosen\nc\n### Rejected\nr\n"
        parsed = parse_text(_doc(body, multi_adapter=True))
        plan = build_plan(parsed)
        assert any(s.type is SectionType.PREFERENCE for s in plan.by_adapter["tone"])


class TestUnknownAdapter:
    def test_unknown_name_rejected(self) -> None:
        body = "::instruction#ghost::\n### Q\nh\n### A\nb\n"
        parsed = parse_text(_doc(body, multi_adapter=True))
        with pytest.raises(UnknownAdapterError, match="ghost"):
            build_plan(parsed)

    def test_sections_for_unknown_name_rejected(self) -> None:
        parsed = parse_text(_doc("prose\n", multi_adapter=True))
        with pytest.raises(UnknownAdapterError, match="unknown_adapter"):
            sections_for(parsed, "unknown_adapter")


class TestSingleAdapterDoc:
    def test_single_adapter_doc_routes_all_to_default(self) -> None:
        body = "# Prose\n\nShared.\n\n::instruction::\n### Q\nh\n### A\nb\n"
        parsed = parse_text(_doc(body, multi_adapter=False))
        plan = build_plan(parsed)
        assert set(plan.by_adapter) == {"default"}
        kinds = {s.type for s in plan.by_adapter["default"]}
        assert SectionType.PROSE in kinds
        assert SectionType.INSTRUCTION in kinds

    def test_single_adapter_doc_rejects_named_section(self) -> None:
        body = "::instruction#tone::\n### Q\nh\n### A\nb\n"
        parsed = parse_text(_doc(body, multi_adapter=False))
        with pytest.raises(UnknownAdapterError, match="tone"):
            build_plan(parsed)


class TestSectionsForShortcut:
    def test_returns_same_as_plan_entry(self) -> None:
        body = "shared prose\n\n::instruction#tone::\n### Q\nh\n### A\nb\n"
        parsed = parse_text(_doc(body, multi_adapter=True))
        plan = build_plan(parsed)
        assert sections_for(parsed, "tone") == plan.by_adapter["tone"]
