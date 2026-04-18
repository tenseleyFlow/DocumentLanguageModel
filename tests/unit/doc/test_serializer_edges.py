"""Edge-case coverage for `dlm.doc.serializer` internals.

Exercises the quoting rules, list formatting, number formatting, and
defensive branches that the higher-level round-trip tests don't reach.
Brings `src/dlm/doc/serializer.py` above the 95% coverage bar called
for by Sprint 03's DoD (audit 02 finding M4).
"""

from __future__ import annotations

import pytest

from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import DlmFrontmatter, TrainingConfig
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import (
    _format_list,
    _format_number,
    _format_string,
    _needs_quoting,
    _scalar,
    _serialize_section,
    serialize,
)

VALID_ULID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"


class TestScalarDispatch:
    def test_bool_true(self) -> None:
        assert _scalar(True) == "true"

    def test_bool_false(self) -> None:
        assert _scalar(False) == "false"

    def test_int_passes_through(self) -> None:
        assert _scalar(42) == "42"

    def test_float_passes_through(self) -> None:
        assert _scalar(0.0) == "0.0"

    def test_str_safe(self) -> None:
        assert _scalar("simple") == "simple"

    def test_list(self) -> None:
        assert _scalar(["a", "b"]) == "[a, b]"

    def test_none(self) -> None:
        assert _scalar(None) == "null"

    def test_unknown_type_uses_str(self) -> None:
        class Weird:
            def __str__(self) -> str:
                return "weird"

        assert _scalar(Weird()) == "weird"


class TestFormatNumber:
    def test_zero_float_is_decimal(self) -> None:
        assert _format_number(0.0) == "0.0"

    def test_int_is_decimal(self) -> None:
        assert _format_number(100) == "100"

    @pytest.mark.parametrize("value", [1e-6, 1e-4, 1e7])
    def test_out_of_range_floats_use_repr(self, value: float) -> None:
        assert _format_number(value) == repr(value)


class TestFormatString:
    def test_empty_string_is_quoted(self) -> None:
        assert _format_string("") == '""'

    def test_safe_string_unquoted(self) -> None:
        assert _format_string("qwen2.5-1.5b") == "qwen2.5-1.5b"

    def test_string_with_whitespace_quoted(self) -> None:
        assert _format_string("has space") == '"has space"'

    def test_string_with_backslash_and_quote_escaped(self) -> None:
        # Input:  a\b"c   →   Output:  "a\\b\"c"
        assert _format_string('a\\b"c has space') == '"a\\\\b\\"c has space"'


class TestNeedsQuoting:
    @pytest.mark.parametrize(
        "reserved",
        ["true", "FALSE", "null", "Yes", "no", "on", "OFF", "~"],
    )
    def test_reserved_scalars_quoted(self, reserved: str) -> None:
        assert _needs_quoting(reserved) is True

    @pytest.mark.parametrize(
        "dangerous",
        [
            "has space",
            "tab\there",
            "line\nbreak",
            "hash #comment",
            "colon:here",
            "quote'marks",
            'double"quote',
            "{brace",
            "]bracket",
        ],
    )
    def test_special_chars_trigger_quoting(self, dangerous: str) -> None:
        assert _needs_quoting(dangerous) is True

    @pytest.mark.parametrize("leading", ["-value", ",value"])
    def test_leading_yaml_markers_trigger_quoting(self, leading: str) -> None:
        assert _needs_quoting(leading) is True

    def test_plain_identifier_not_quoted(self) -> None:
        assert _needs_quoting("adamw_torch") is False


class TestFormatList:
    def test_empty_list(self) -> None:
        assert _format_list([]) == "[]"

    def test_string_items(self) -> None:
        assert _format_list(["q_proj", "v_proj"]) == "[q_proj, v_proj]"

    def test_mixed_types(self) -> None:
        assert _format_list([1, "x", True, None]) == "[1, x, true, null]"


class TestSerializeSection:
    def test_prose_without_trailing_newline_gets_one(self) -> None:
        s = Section(SectionType.PROSE, "no newline")
        assert _serialize_section(s) == "no newline\n"

    def test_prose_with_trailing_newline_unchanged(self) -> None:
        s = Section(SectionType.PROSE, "has newline\n")
        assert _serialize_section(s) == "has newline\n"

    def test_fenced_section_emits_fence_and_body(self) -> None:
        s = Section(SectionType.INSTRUCTION, "### Q\nquestion\n")
        out = _serialize_section(s)
        assert out.startswith("::instruction::\n")
        assert out.endswith("\n")
        assert "question" in out

    def test_fenced_empty_content_is_just_fence(self) -> None:
        s = Section(SectionType.PREFERENCE, "")
        assert _serialize_section(s) == "::preference::\n"

    def test_fenced_content_without_trailing_newline_gets_one(self) -> None:
        s = Section(SectionType.INSTRUCTION, "raw")
        assert _serialize_section(s) == "::instruction::\nraw\n"


class TestSerializeTrailingNewline:
    def test_output_always_ends_with_single_newline(self) -> None:
        """Covers the `if not rendered.endswith('\\n')` branch in serialize()."""
        fm = DlmFrontmatter(dlm_id=VALID_ULID, base_model="smollm2-135m")
        # Manually construct a section whose content has no trailing newline;
        # the section serializer already normalizes, so top-level tail
        # normalization is the belt-and-braces branch under test.
        parsed = ParsedDlm(
            frontmatter=fm,
            sections=(Section(SectionType.PROSE, "content"),),
        )
        out = serialize(parsed)
        assert out.endswith("\n")
        assert not out.endswith("\n\n")


class TestFrontmatterExplicitTargetModulesList:
    """Ensures the list branch in the nested-mapping emitter is exercised."""

    def test_list_value_renders_as_flow_sequence(self) -> None:
        fm = DlmFrontmatter(
            dlm_id=VALID_ULID,
            base_model="smollm2-135m",
            training=TrainingConfig(target_modules=["q_proj", "v_proj"]),
        )
        parsed = ParsedDlm(frontmatter=fm, sections=())
        out = serialize(parsed)
        assert "target_modules: [q_proj, v_proj]" in out

    def test_system_prompt_emits_block_scalar(self) -> None:
        fm = DlmFrontmatter(
            dlm_id=VALID_ULID,
            base_model="smollm2-135m",
            system_prompt="first line\nsecond line",
        )
        parsed = ParsedDlm(frontmatter=fm, sections=())
        out = serialize(parsed)
        assert "system_prompt: |\n  first line\n  second line" in out
