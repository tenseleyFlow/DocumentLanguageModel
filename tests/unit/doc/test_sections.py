"""Section content-hash ID semantics and immutability."""

from __future__ import annotations

import dataclasses

import pytest

from dlm.doc.sections import Section, SectionType


class TestSectionId:
    def test_same_type_same_content_same_id(self) -> None:
        a = Section(SectionType.PROSE, "hello\n")
        b = Section(SectionType.PROSE, "hello\n")
        assert a.section_id == b.section_id
        assert len(a.section_id) == 16

    def test_different_type_different_id(self) -> None:
        content = "identical body\n"
        prose = Section(SectionType.PROSE, content)
        instr = Section(SectionType.INSTRUCTION, content)
        assert prose.section_id != instr.section_id

    def test_different_content_different_id(self) -> None:
        a = Section(SectionType.PROSE, "one\n")
        b = Section(SectionType.PROSE, "two\n")
        assert a.section_id != b.section_id

    def test_crlf_and_lf_have_same_id(self) -> None:
        """Audit F15: section IDs must be stable across line-ending edits."""
        lf = Section(SectionType.PROSE, "line1\nline2\n")
        crlf = Section(SectionType.PROSE, "line1\r\nline2\r\n")
        assert lf.section_id == crlf.section_id

    def test_bom_stripped_for_hashing(self) -> None:
        """Leading BOM must not change the section ID."""
        with_bom = Section(SectionType.PROSE, "\ufeffbody\n")
        without_bom = Section(SectionType.PROSE, "body\n")
        assert with_bom.section_id == without_bom.section_id

    def test_start_line_not_part_of_identity(self) -> None:
        a = Section(SectionType.PROSE, "body\n", start_line=5)
        b = Section(SectionType.PROSE, "body\n", start_line=500)
        assert a.section_id == b.section_id


class TestSectionImmutability:
    def test_is_frozen(self) -> None:
        s = Section(SectionType.PROSE, "body\n")
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.content = "other"  # type: ignore[misc]

    def test_equality_based_on_fields(self) -> None:
        a = Section(SectionType.PROSE, "body\n", start_line=1)
        b = Section(SectionType.PROSE, "body\n", start_line=1)
        assert a == b

    def test_inequality_when_start_line_differs(self) -> None:
        a = Section(SectionType.PROSE, "body\n", start_line=1)
        b = Section(SectionType.PROSE, "body\n", start_line=2)
        assert a != b  # full struct equality; section_id equality is weaker
