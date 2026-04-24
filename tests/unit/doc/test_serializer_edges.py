"""Edge-case coverage for `dlm.doc.serializer` internals.

Exercises the quoting rules, list formatting, number formatting, and
defensive branches that the higher-level round-trip tests don't reach.
Brings `src/dlm/doc/serializer.py` above the 95% coverage bar called
for by Sprint 03's DoD (audit 02 finding M4).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from dlm.doc.parser import ParsedDlm, parse_text
from dlm.doc.schema import (
    DlmFrontmatter,
    PreferenceConfig,
    PreferenceHyperparams,
    TrainingConfig,
)
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import (
    _emit_nested_mapping,
    _format_list,
    _format_number,
    _format_string,
    _marker_attr_value,
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
    def test_output_always_ends_with_single_newline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Covers the `if not rendered.endswith('\\n')` branch in serialize()."""
        fm = DlmFrontmatter(dlm_id=VALID_ULID, base_model="smollm2-135m")
        parsed = ParsedDlm(
            frontmatter=fm,
            sections=(Section(SectionType.PROSE, "content"),),
        )
        monkeypatch.setattr("dlm.doc.serializer._serialize_frontmatter", lambda _fm: "---\n---")
        out = serialize(parsed)
        assert out.endswith("\n")
        assert not out.endswith("\n\n")

    def test_serializer_adds_newline_when_section_render_omits_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fm = DlmFrontmatter(dlm_id=VALID_ULID, base_model="smollm2-135m")
        parsed = ParsedDlm(
            frontmatter=fm,
            sections=(Section(SectionType.PROSE, "content"),),
        )
        monkeypatch.setattr("dlm.doc.serializer._serialize_frontmatter", lambda _fm: "---\n---")
        monkeypatch.setattr("dlm.doc.serializer._serialize_section", lambda _section: "body")
        assert serialize(parsed).endswith("\n")


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


class TestPreferenceNestedBlock:
    """`training.preference` is a pydantic model containing its own
    nested `hyperparams` model. Default-equal instances are suppressed;
    non-defaults render as nested sub-blocks with 4-space indents."""

    def test_all_default_preference_suppressed(self) -> None:
        fm = DlmFrontmatter(
            dlm_id=VALID_ULID,
            base_model="smollm2-135m",
            training=TrainingConfig(),
        )
        parsed = ParsedDlm(frontmatter=fm, sections=())
        out = serialize(parsed)
        assert "preference:" not in out
        assert "dpo:" not in out

    def test_non_default_preference_emits_sub_block(self) -> None:
        fm = DlmFrontmatter(
            dlm_id=VALID_ULID,
            base_model="smollm2-135m",
            training=TrainingConfig(
                preference=PreferenceConfig(
                    enabled=True,
                    hyperparams=PreferenceHyperparams(beta=0.2),
                ),
            ),
        )
        parsed = ParsedDlm(frontmatter=fm, sections=())
        out = serialize(parsed)
        assert "training:\n" in out
        assert "  preference:\n" in out
        assert "    enabled: true" in out
        assert "    hyperparams:\n" in out
        assert "      beta:" in out
        # still-default fields (method, loss_type, reference) are suppressed.
        # Match on `key:` form to avoid false hits inside `preference`.
        assert "method:" not in out
        assert "loss_type:" not in out
        assert "    reference:" not in out

    def test_round_trip_preserves_preference_overrides(self) -> None:
        source = (
            "---\n"
            f"dlm_id: {VALID_ULID}\n"
            "base_model: smollm2-135m\n"
            "training:\n"
            "  preference:\n"
            "    enabled: true\n"
            "    loss_type: ipo\n"
            "    hyperparams:\n"
            "      beta: 0.25\n"
            "---\n"
        )
        parsed = parse_text(source)
        assert parsed.frontmatter.training.preference.enabled is True
        assert parsed.frontmatter.training.preference.hyperparams.beta == 0.25
        assert parsed.frontmatter.training.preference.loss_type == "ipo"
        # Idempotency contract: pipeline twice == pipeline once.
        rendered = serialize(parsed)
        assert serialize(parse_text(rendered)) == rendered


class TestNestedModelCoverage:
    def test_explicit_all_default_nested_model_is_suppressed(self) -> None:
        class Child(BaseModel):
            enabled: bool = False

        class Parent(BaseModel):
            child: Child

        assert _emit_nested_mapping(Parent(child=Child()), indent=2) == []

    def test_list_of_models_renders_nested_yaml_items(self) -> None:
        class Item(BaseModel):
            name: str
            enabled: bool = False

        class Parent(BaseModel):
            items: list[Item]

        lines = _emit_nested_mapping(
            Parent(items=[Item(name="docs"), Item(name="blog", enabled=True)]),
            indent=2,
        )
        assert lines == [
            "  items:",
            "    - name: docs",
            "    - name: blog",
            "      enabled: true",
        ]

    def test_list_of_all_default_models_emits_empty_mapping_item(self) -> None:
        class Item(BaseModel):
            enabled: bool = False

        class Parent(BaseModel):
            items: list[Item]

        lines = _emit_nested_mapping(Parent(items=[Item()]), indent=2)
        assert lines == ["  items:", "    - {}"]


class TestMetadataMarkerCoverage:
    def test_auto_mined_requires_full_metadata_at_serialize_time(self) -> None:
        section = Section(
            SectionType.PREFERENCE,
            "### Prompt\np\n### Chosen\nc\n### Rejected\nr\n",
            auto_mined=True,
            judge_name="sway",
            judge_score_chosen=1.0,
            judge_score_rejected=0.0,
            mined_at="2026-04-24T00:00:00Z",
            mined_run_id=1,
        )
        object.__setattr__(section, "judge_name", None)
        with pytest.raises(ValueError, match="missing required metadata"):
            _serialize_section(section)

    def test_auto_synth_requires_full_metadata_at_serialize_time(self) -> None:
        section = Section(
            SectionType.INSTRUCTION,
            "### Q\nq\n### A\na\n",
            auto_synth=True,
            synth_teacher="self",
            synth_strategy="extraction",
            synth_at="2026-04-24T00:00:00Z",
            source_section_id="0123456789abcdef",
        )
        object.__setattr__(section, "synth_teacher", None)
        with pytest.raises(ValueError, match="missing required metadata"):
            _serialize_section(section)

    def test_marker_values_reject_unroundtrippable_content(self) -> None:
        with pytest.raises(ValueError, match="cannot contain double-quotes or newlines"):
            _marker_attr_value('bad"value')


class TestTrainingConfigCoverage:
    def test_explicit_none_adapters_is_allowed(self) -> None:
        training = TrainingConfig(adapters=None)
        assert training.adapters is None
