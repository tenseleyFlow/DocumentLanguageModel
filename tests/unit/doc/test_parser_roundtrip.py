"""Parser + serializer integration: parse, tokenize, round-trip, encoding."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.doc.errors import (
    DlmVersionError,
    FenceError,
    FrontmatterError,
    SchemaValidationError,
)
from dlm.doc.parser import ParsedDlm, parse_file, parse_text
from dlm.doc.sections import SectionType
from dlm.doc.serializer import serialize
from dlm.io.text import DlmEncodingError
from tests.fixtures.dlm_factory import instruction, make_dlm, preference, prose

VALID_ULID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"


class TestParseValidDocuments:
    def test_factory_default_parses(self) -> None:
        text = make_dlm(dlm_id=VALID_ULID)
        parsed = parse_text(text)
        assert isinstance(parsed, ParsedDlm)
        assert parsed.frontmatter.dlm_id == VALID_ULID
        assert parsed.frontmatter.base_model == "smollm2-135m"
        assert len(parsed.sections) >= 1

    def test_prose_only_document(self) -> None:
        text = make_dlm(
            dlm_id=VALID_ULID,
            sections=[prose("# intro\n\nJust text here.\n")],
        )
        parsed = parse_text(text)
        assert len(parsed.sections) == 1
        assert parsed.sections[0].type == SectionType.PROSE
        assert "Just text here." in parsed.sections[0].content

    def test_instruction_section_roundtrips_content(self) -> None:
        text = make_dlm(
            dlm_id=VALID_ULID,
            sections=[instruction(("Q1?", "A1."), ("Q2?", "A2."))],
        )
        parsed = parse_text(text)
        assert len(parsed.sections) == 1
        assert parsed.sections[0].type == SectionType.INSTRUCTION

    def test_preference_section_accepted_in_v1(self) -> None:
        text = make_dlm(
            dlm_id=VALID_ULID,
            sections=[preference(("p", "c", "r"))],
        )
        parsed = parse_text(text)
        assert len(parsed.sections) == 1
        assert parsed.sections[0].type == SectionType.PREFERENCE

    def test_mixed_sections_preserve_order(self) -> None:
        text = make_dlm(
            dlm_id=VALID_ULID,
            sections=[
                prose("intro.\n"),
                instruction(("q", "a")),
                preference(("p", "c", "r")),
            ],
        )
        parsed = parse_text(text)
        types = [s.type for s in parsed.sections]
        assert types == [
            SectionType.PROSE,
            SectionType.INSTRUCTION,
            SectionType.PREFERENCE,
        ]


class TestFrontmatterErrors:
    def test_missing_opening_delimiter(self) -> None:
        text = "no frontmatter here\n"
        with pytest.raises(FrontmatterError) as exc:
            parse_text(text)
        assert exc.value.line == 1

    def test_missing_closing_delimiter(self) -> None:
        text = "---\ndlm_id: " + VALID_ULID + "\n"
        with pytest.raises(FrontmatterError, match="no closing"):
            parse_text(text)

    def test_non_mapping_yaml(self) -> None:
        text = "---\n- just\n- a\n- list\n---\n"
        with pytest.raises(FrontmatterError, match="must be a mapping"):
            parse_text(text)

    def test_unknown_top_level_key_reports_schema_error(self) -> None:
        text = f"---\ndlm_id: {VALID_ULID}\nbase_model: smollm2-135m\nsurprise: 1\n---\n"
        with pytest.raises(SchemaValidationError, match="surprise"):
            parse_text(text)

    def test_invalid_yaml_surfaces_location(self) -> None:
        text = "---\ndlm_id: [unclosed\n---\n"
        with pytest.raises(FrontmatterError, match="invalid YAML"):
            parse_text(text)


class TestVersionGating:
    def test_future_version_refused(self) -> None:
        text = f"---\ndlm_id: {VALID_ULID}\nbase_model: smollm2-135m\ndlm_version: 999\n---\n"
        with pytest.raises(DlmVersionError, match="newer than this parser"):
            parse_text(text)

    def test_sub_current_version_without_migrator_refuses(self) -> None:
        """Sprint 12b: sub-CURRENT documents route through the migration
        dispatcher; a gap in the `MIGRATORS` registry raises
        `UnsupportedMigrationError` (subclass of `DlmVersionError`).

        Simulate CURRENT=2 with no v1 migrator registered; the dispatcher
        refuses to silently accept a v1 dict. The coverage test in
        `tests/unit/doc/test_migrations.py` is the static enforcement;
        this is the runtime gate.
        """
        from dlm.doc import versioned as versioned_module
        from dlm.doc.errors import UnsupportedMigrationError

        text = f"---\ndlm_id: {VALID_ULID}\nbase_model: smollm2-135m\ndlm_version: 1\n---\n"
        original = versioned_module.CURRENT_SCHEMA_VERSION
        versioned_module.CURRENT_SCHEMA_VERSION = original + 1
        try:
            with pytest.raises(UnsupportedMigrationError, match="no migrator"):
                parse_text(text)
        finally:
            versioned_module.CURRENT_SCHEMA_VERSION = original


class TestFenceGrammar:
    def test_unknown_fence_raises(self) -> None:
        text = make_dlm(
            dlm_id=VALID_ULID,
            sections=[prose("before fence\n")],
        )
        # Append a bogus fence after parsing-friendly content.
        text += "\n::weird::\n\nbody\n"
        with pytest.raises(FenceError, match="unknown section fence"):
            parse_text(text)

    def test_fence_inside_code_block_is_literal(self) -> None:
        body = (
            "Some prose.\n"
            "\n"
            "```\n"
            "::instruction::\n"  # should NOT be parsed as a fence
            "```\n"
        )
        text = make_dlm(dlm_id=VALID_ULID, sections=[prose(body)])
        parsed = parse_text(text)
        assert len(parsed.sections) == 1
        assert parsed.sections[0].type == SectionType.PROSE

    def test_unterminated_code_block_raises(self) -> None:
        # Author the text manually so the factory doesn't terminate it.
        text = f"---\ndlm_id: {VALID_ULID}\nbase_model: smollm2-135m\n---\n\n```\nforever open\n"
        with pytest.raises(FenceError, match="unterminated"):
            parse_text(text)


class TestParsedDlmImmutability:
    def test_sections_is_tuple(self) -> None:
        parsed = parse_text(make_dlm(dlm_id=VALID_ULID))
        assert isinstance(parsed.sections, tuple)


class TestRoundTrip:
    def test_parse_then_serialize_then_parse_matches(self) -> None:
        original = make_dlm(
            dlm_id=VALID_ULID,
            sections=[
                prose("# Heading\n\nParagraph.\n"),
                instruction(("What is 2+2?", "4."), ("And 3+3?", "6.")),
                preference(("Greet.", "Hi!", "hey.")),
            ],
            training_overrides={"lora_r": 16, "num_epochs": 2},
            system_prompt="You are helpful.\nAlways be concise.",
        )
        first = parse_text(original)
        serialized = serialize(first)
        second = parse_text(serialized)
        # Frontmatter equality
        assert first.frontmatter == second.frontmatter
        # Section types + content equality
        assert [s.type for s in first.sections] == [s.type for s in second.sections]
        assert [s.content for s in first.sections] == [s.content for s in second.sections]

    def test_second_pass_is_byte_identical(self) -> None:
        original = make_dlm(
            dlm_id=VALID_ULID,
            sections=[prose("content\n"), instruction(("q", "a"))],
        )
        once = serialize(parse_text(original))
        twice = serialize(parse_text(once))
        assert once == twice  # idempotent


class TestSectionIdStabilityUnderEdits:
    """Whitespace-only edits in one section must not change another section's ID."""

    def test_editing_one_prose_does_not_change_other_ids(self) -> None:
        doc_before = make_dlm(
            dlm_id=VALID_ULID,
            sections=[
                prose("first section\n"),
                instruction(("stable q", "stable a")),
            ],
        )
        doc_after = make_dlm(
            dlm_id=VALID_ULID,
            sections=[
                prose("first section   \n"),  # trailing spaces added
                instruction(("stable q", "stable a")),
            ],
        )
        a = parse_text(doc_before).sections[1]
        b = parse_text(doc_after).sections[1]
        assert a.section_id == b.section_id


class TestEncodingContract:
    """Audit F15: UTF-8 strict, BOM strip, CRLF normalization."""

    def test_parse_file_strips_bom(self, tmp_path: Path) -> None:
        text = make_dlm(dlm_id=VALID_ULID)
        p = tmp_path / "with_bom.dlm"
        p.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
        parsed = parse_file(p)
        assert parsed.frontmatter.dlm_id == VALID_ULID

    def test_parse_file_normalizes_crlf(self, tmp_path: Path) -> None:
        text = make_dlm(dlm_id=VALID_ULID)
        p = tmp_path / "crlf.dlm"
        p.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
        parsed = parse_file(p)
        assert parsed.frontmatter.dlm_id == VALID_ULID

    def test_invalid_utf8_raises_encoding_error(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.dlm"
        p.write_bytes(b"---\nbase_model: \xff\n---\n")
        with pytest.raises(DlmEncodingError) as exc:
            parse_file(p)
        # Audit-02 minor: the parser path must surface byte_offset
        # end-to-end (not just the io-layer test).
        assert exc.value.byte_offset == len("---\nbase_model: ")
        assert exc.value.path == p

    def test_crlf_and_lf_produce_identical_section_ids(self, tmp_path: Path) -> None:
        """Windows and Unix edits of the same content must hash-identically."""
        text = make_dlm(
            dlm_id=VALID_ULID,
            sections=[prose("body line 1\nbody line 2\n")],
        )
        lf_path = tmp_path / "lf.dlm"
        lf_path.write_bytes(text.encode("utf-8"))
        crlf_path = tmp_path / "crlf.dlm"
        crlf_path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
        lf_parsed = parse_file(lf_path)
        crlf_parsed = parse_file(crlf_path)
        lf_ids = [s.section_id for s in lf_parsed.sections]
        crlf_ids = [s.section_id for s in crlf_parsed.sections]
        assert lf_ids == crlf_ids
