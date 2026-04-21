"""Schema v7: `Section.auto_harvest` + `harvest_source` with magic-comment
round-trip (Sprint 33.1).

Non-PROSE sections carry the harvest metadata as a first-line magic
comment. The parser lifts it into `Section` fields; the serializer
emits it back. `section_id` ignores the new fields — two sections with
identical content but different harvest provenance share a content
hash, matching the "metadata churn doesn't invalidate replay" rule.
"""

from __future__ import annotations

from dlm.doc.parser import parse_text
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import serialize

_FRONTMATTER = """---
dlm_id: 01KPQ9X1000000000000000000
dlm_version: 8
base_model: smollm2-135m
---
"""


def _parsed(body: str):
    return parse_text(_FRONTMATTER + body)


class TestHarvestMarkerParse:
    def test_marker_lifts_into_fields(self) -> None:
        body = (
            "::instruction::\n"
            '<!-- dlm-auto-harvest: source="run_7_sway" -->\n'
            "### Q\nwhat is a quine?\n### A\na program that prints itself.\n"
        )
        parsed = _parsed(body)
        assert len(parsed.sections) == 1
        section = parsed.sections[0]
        assert section.auto_harvest is True
        assert section.harvest_source == "run_7_sway"
        # marker is stripped from content
        assert "<!--" not in section.content
        assert "### Q" in section.content

    def test_no_marker_defaults(self) -> None:
        body = "::instruction::\n### Q\nhi?\n### A\nhello.\n"
        parsed = _parsed(body)
        section = parsed.sections[0]
        assert section.auto_harvest is False
        assert section.harvest_source is None

    def test_marker_in_prose_is_literal(self) -> None:
        """PROSE bodies are whatever the user writes — the magic comment
        should not be interpreted there."""
        body = '<!-- dlm-auto-harvest: source="fake" -->\nthis is prose.\n'
        parsed = _parsed(body)
        assert len(parsed.sections) == 1
        section = parsed.sections[0]
        assert section.type == SectionType.PROSE
        assert section.auto_harvest is False
        assert "<!--" in section.content


class TestHarvestMarkerSerialize:
    def test_roundtrip_preserves_marker(self) -> None:
        body = (
            "::instruction::\n"
            '<!-- dlm-auto-harvest: source="run_7_sway" -->\n'
            "### Q\nping?\n### A\npong.\n"
        )
        once = serialize(_parsed(body))
        twice = serialize(_parsed(once.split("---\n", 2)[-1]))
        # idempotent after the frontmatter stabilizes
        assert once == _FRONTMATTER + body or twice == once

    def test_no_marker_emitted_for_normal_sections(self) -> None:
        body = "::instruction::\n### Q\nhi?\n### A\nhello.\n"
        out = serialize(_parsed(body))
        assert "dlm-auto-harvest" not in out


class TestHarvestIdentity:
    def test_section_id_ignores_harvest_fields(self) -> None:
        """Provenance is metadata, not identity — a harvested section
        and a hand-authored section with identical content share a
        `section_id`."""
        harvested = Section(
            type=SectionType.INSTRUCTION,
            content="### Q\nsame body.\n### A\nyes.",
            auto_harvest=True,
            harvest_source="run_7_sway",
        )
        authored = Section(
            type=SectionType.INSTRUCTION,
            content="### Q\nsame body.\n### A\nyes.",
        )
        assert harvested.section_id == authored.section_id


class TestSchemaVersionBump:
    def test_parser_accepts_current(self) -> None:
        from dlm.doc.schema import CURRENT_SCHEMA_VERSION

        parsed = _parsed("::instruction::\n### Q\nhi?\n### A\nhello.\n")
        assert parsed.frontmatter.dlm_version == CURRENT_SCHEMA_VERSION

    def test_v6_document_still_parses(self) -> None:
        """A v6 document without the new fields parses cleanly under v7
        — the migrator is pure identity (v6.py registers from_version=6
        identity, chained from older via the registry)."""
        body = (
            "---\n"
            "dlm_id: 01KPQ9X1000000000000000000\n"
            "dlm_version: 6\n"
            "base_model: smollm2-135m\n"
            "---\n"
            "::instruction::\n### Q\nhi?\n### A\nhello.\n"
        )
        parsed = parse_text(body)
        assert len(parsed.sections) == 1
        assert parsed.sections[0].auto_harvest is False
