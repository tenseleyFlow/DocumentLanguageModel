"""Fence grammar: `::<type>::` and `::<type>#<adapter>::`.

Covers the parser extension, serializer round-trip, and validation
of the adapter-name suffix against the shared grammar.
"""

from __future__ import annotations

from textwrap import dedent

import pytest

from dlm.doc.errors import FenceError
from dlm.doc.parser import parse_text
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import serialize

_FM = dedent(
    """\
    ---
    dlm_id: 01HZ4X7TGZM3J1A2B3C4D5E6F7
    base_model: smollm2-135m
    ---
    """
)


def _parse(body: str):  # type: ignore[no-untyped-def]
    return parse_text(_FM + body)


class TestParseFenceSuffix:
    def test_bare_fence_has_no_adapter(self) -> None:
        parsed = _parse("::instruction::\n### Q\nhi\n### A\nbye\n")
        instr = [s for s in parsed.sections if s.type == SectionType.INSTRUCTION]
        assert instr
        assert instr[0].adapter is None

    def test_suffixed_fence_captures_adapter(self) -> None:
        parsed = _parse("::instruction#tone::\n### Q\nhi\n### A\nbye\n")
        instr = [s for s in parsed.sections if s.type == SectionType.INSTRUCTION]
        assert instr
        assert instr[0].adapter == "tone"

    def test_preference_fence_adapter(self) -> None:
        parsed = _parse(
            "::preference#knowledge::\n### Prompt\nq\n### Chosen\nc\n### Rejected\nr\n"
        )
        pref = [s for s in parsed.sections if s.type == SectionType.PREFERENCE]
        assert pref
        assert pref[0].adapter == "knowledge"

    def test_empty_suffix_after_hash_rejected(self) -> None:
        with pytest.raises(FenceError, match="empty adapter suffix"):
            _parse("::instruction#::\n### Q\nhi\n### A\nbye\n")

    @pytest.mark.parametrize(
        "adapter_name",
        # Names with chars the broader fence regex accepts (alphanumeric,
        # underscore, hyphen, case) but our grammar forbids. Names with
        # spaces or dots don't even match `_FENCE_RE`, so they fall
        # through as prose — not an adapter-suffix error.
        ["Tone", "1tone", "tone-name", "_tone"],
    )
    def test_invalid_suffix_rejected(self, adapter_name: str) -> None:
        with pytest.raises(FenceError, match="invalid adapter name"):
            _parse(f"::instruction#{adapter_name}::\n### Q\nh\n### A\nb\n")

    def test_unknown_base_type_still_errors(self) -> None:
        with pytest.raises(FenceError, match="unknown section fence"):
            _parse("::mystery#tone::\nbody\n")


class TestSerializeRoundTrip:
    def test_serialize_emits_suffix(self) -> None:
        body = "::instruction#tone::\n### Q\nhi\n### A\nbye\n"
        parsed = _parse(body)
        out = serialize(parsed)
        assert "::instruction#tone::" in out

    def test_serialize_omits_suffix_when_none(self) -> None:
        body = "::instruction::\n### Q\nhi\n### A\nbye\n"
        parsed = _parse(body)
        out = serialize(parsed)
        assert "::instruction::" in out
        assert "#" not in out.split("::instruction")[1].split("::")[0]

    def test_double_round_trip_is_idempotent(self) -> None:
        body = "::instruction#knowledge::\n### Q\nQ1\n### A\nA1\n"
        once = serialize(_parse(body))
        twice = serialize(parse_text(once))
        assert once == twice


class TestSectionIdentityUnchanged:
    def test_adapter_not_in_section_id(self) -> None:
        """Routing is structural, not content — same content with and
        without a `#adapter` suffix must produce the same section_id
        so replay snapshots don't duplicate rows on routing edits."""
        s_plain = Section(
            type=SectionType.INSTRUCTION, content="### Q\nhi\n### A\nbye"
        )
        s_routed = Section(
            type=SectionType.INSTRUCTION,
            content="### Q\nhi\n### A\nbye",
            adapter="tone",
        )
        assert s_plain.section_id == s_routed.section_id
