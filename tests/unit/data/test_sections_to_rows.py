"""`sections_to_rows` maps each SectionType to the right row shape."""

from __future__ import annotations

import pytest

from dlm.data.errors import InstructionParseError
from dlm.data.sections_to_rows import sections_to_rows
from dlm.doc.sections import Section, SectionType


def _s(section_type: SectionType, content: str) -> Section:
    return Section(type=section_type, content=content, start_line=1)


class TestProseShape:
    def test_prose_becomes_text_row(self) -> None:
        s = _s(SectionType.PROSE, "This is prose.")
        rows = sections_to_rows([s])
        assert len(rows) == 1
        assert rows[0]["text"] == "This is prose."
        assert rows[0]["_dlm_section_id"] == s.section_id

    def test_empty_prose_dropped(self) -> None:
        s = _s(SectionType.PROSE, "   \n\n  ")
        assert sections_to_rows([s]) == []

    def test_prose_content_stripped(self) -> None:
        s = _s(SectionType.PROSE, "\n\nhello\n\n")
        rows = sections_to_rows([s])
        assert rows[0]["text"] == "hello"


class TestInstructionShape:
    def test_each_qa_pair_becomes_messages_row(self) -> None:
        body = "### Q\nq1\n### A\na1\n\n### Q\nq2\n### A\na2"
        s = _s(SectionType.INSTRUCTION, body)
        rows = sections_to_rows([s])
        assert len(rows) == 2
        assert rows[0]["messages"] == [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        assert rows[1]["messages"][1]["content"] == "a2"
        # Both rows share the same section id — splitter uses sub-index.
        assert rows[0]["_dlm_section_id"] == rows[1]["_dlm_section_id"]

    def test_malformed_instruction_bubbles_up(self) -> None:
        s = _s(SectionType.INSTRUCTION, "### Q\nunterm")
        with pytest.raises(InstructionParseError):
            sections_to_rows([s])

    def test_probe_markers_normalized_before_parse(self) -> None:
        s = _s(SectionType.INSTRUCTION, "### Q !probe\nq1\n### A\na1")
        rows = sections_to_rows([s])
        assert rows == [
            {
                "messages": [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                ],
                "_dlm_section_id": s.section_id,
                "_dlm_row_tags": {},
            },
        ]


class TestPreferenceShape:
    def test_each_triple_becomes_preference_row(self) -> None:
        body = (
            "### Prompt\np1\n### Chosen\nc1\n### Rejected\nr1\n\n"
            "### Prompt\np2\n### Chosen\nc2\n### Rejected\nr2"
        )
        s = _s(SectionType.PREFERENCE, body)
        rows = sections_to_rows([s])
        assert len(rows) == 2
        assert rows[0]["prompt"] == "p1"
        assert rows[0]["chosen"] == "c1"
        assert rows[0]["rejected"] == "r1"
        assert rows[1]["prompt"] == "p2"


class TestOrderPreservation:
    def test_insertion_order_preserved_across_types(self) -> None:
        a = _s(SectionType.PROSE, "prose1")
        b = _s(SectionType.INSTRUCTION, "### Q\nq\n### A\na")
        c = _s(SectionType.PROSE, "prose2")
        rows = sections_to_rows([a, b, c])
        assert rows[0]["text"] == "prose1"
        assert "messages" in rows[1]
        assert rows[2]["text"] == "prose2"
