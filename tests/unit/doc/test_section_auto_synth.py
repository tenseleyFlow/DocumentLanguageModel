"""Schema v15: auto-synth instruction metadata marker round-trip."""

from __future__ import annotations

import pytest

from dlm.doc.errors import FenceError
from dlm.doc.parser import parse_text
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import serialize

_FRONTMATTER = """---
dlm_id: 01KPQ9X1000000000000000000
dlm_version: 15
base_model: smollm2-135m
---
"""


def _parsed(body: str):
    return parse_text(_FRONTMATTER + body)


_AUTO_SYNTH = (
    '<!-- dlm-auto-synth: synth_teacher="self" '
    'synth_strategy="extraction" '
    'synth_at="2026-04-23T20:00:00Z" '
    'source_section_id="0123abc456def789" -->\n'
)


class TestAutoSynthMarkerParse:
    def test_marker_lifts_into_fields(self) -> None:
        body = "::instruction::\n" + _AUTO_SYNTH + "### Q\nhi?\n### A\nhello.\n"
        parsed = _parsed(body)
        assert len(parsed.sections) == 1
        section = parsed.sections[0]
        assert section.auto_synth is True
        assert section.synth_teacher == "self"
        assert section.synth_strategy == "extraction"
        assert section.synth_at == "2026-04-23T20:00:00Z"
        assert section.source_section_id == "0123abc456def789"
        assert "dlm-auto-synth" not in section.content

    def test_marker_in_non_instruction_is_literal(self) -> None:
        body = (
            "::preference::\n" + _AUTO_SYNTH + "### Prompt\nhi?\n### Chosen\na\n### Rejected\nb\n"
        )
        parsed = _parsed(body)
        section = parsed.sections[0]
        assert section.type == SectionType.PREFERENCE
        assert section.auto_synth is False
        assert "dlm-auto-synth" in section.content

    def test_bad_source_section_id_refused(self) -> None:
        body = (
            "::instruction::\n"
            '<!-- dlm-auto-synth: synth_teacher="self" '
            'synth_strategy="extraction" '
            'synth_at="2026-04-23T20:00:00Z" '
            'source_section_id="NOT-HEX" -->\n'
            "### Q\nhi?\n### A\nhello.\n"
        )
        with pytest.raises(FenceError, match="source_section_id must be a 16-char lowercase hex"):
            _parsed(body)


class TestAutoSynthMarkerSerialize:
    def test_roundtrip_emits_marker(self) -> None:
        doc = _parsed("::instruction::\n### Q\nx\n### A\ny\n")
        section = Section(
            type=SectionType.INSTRUCTION,
            content="### Q\nquestion?\n### A\nanswer.\n",
            auto_synth=True,
            synth_teacher="vllm-server:http://localhost:8000",
            synth_strategy="expansion",
            synth_at="2026-04-23T20:00:00Z",
            source_section_id="0123abc456def789",
        )
        updated = type(doc)(
            frontmatter=doc.frontmatter,
            sections=(section,),
            source_path=doc.source_path,
        )
        out = serialize(updated)
        reparsed = parse_text(out)
        synth = reparsed.sections[0]
        assert synth.auto_synth is True
        assert synth.synth_teacher == "vllm-server:http://localhost:8000"
        assert synth.synth_strategy == "expansion"
        assert synth.source_section_id == "0123abc456def789"

    def test_missing_metadata_refused(self) -> None:
        with pytest.raises(ValueError, match="require metadata fields"):
            Section(
                type=SectionType.INSTRUCTION,
                content="### Q\nquestion?\n### A\nanswer.\n",
                auto_synth=True,
                synth_teacher="self",
                synth_strategy="extraction",
                synth_at="2026-04-23T20:00:00Z",
            )


class TestAutoSynthIdentity:
    def test_section_id_ignores_auto_synth_fields(self) -> None:
        synth = Section(
            type=SectionType.INSTRUCTION,
            content="### Q\nsame\n### A\nanswer\n",
            auto_synth=True,
            synth_teacher="self",
            synth_strategy="both",
            synth_at="2026-04-23T20:00:00Z",
            source_section_id="0123abc456def789",
        )
        authored = Section(
            type=SectionType.INSTRUCTION,
            content="### Q\nsame\n### A\nanswer\n",
        )
        assert synth.section_id == authored.section_id
