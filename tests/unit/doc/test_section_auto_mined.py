"""Schema v14: auto-mined preference metadata marker round-trip."""

from __future__ import annotations

import pytest

from dlm.doc.errors import FenceError
from dlm.doc.parser import parse_text
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import serialize

_FRONTMATTER = """---
dlm_id: 01KPQ9X1000000000000000000
dlm_version: 14
base_model: smollm2-135m
---
"""


def _parsed(body: str):
    return parse_text(_FRONTMATTER + body)


_AUTO_MINED = (
    '<!-- dlm-auto-mined: judge_name="sway:preference_judge" '
    'judge_score_chosen="0.9" judge_score_rejected="0.1" '
    'mined_at="2026-04-23T20:00:00Z" mined_run_id="7" -->\n'
)


class TestAutoMinedMarkerParse:
    def test_marker_lifts_into_fields(self) -> None:
        body = (
            "::preference::\n"
            + _AUTO_MINED
            + "### Prompt\nquestion?\n### Chosen\nbetter\n### Rejected\nworse\n"
        )
        parsed = _parsed(body)
        assert len(parsed.sections) == 1
        section = parsed.sections[0]
        assert section.auto_mined is True
        assert section.judge_name == "sway:preference_judge"
        assert section.judge_score_chosen == pytest.approx(0.9)
        assert section.judge_score_rejected == pytest.approx(0.1)
        assert section.mined_at == "2026-04-23T20:00:00Z"
        assert section.mined_run_id == 7
        assert "dlm-auto-mined" not in section.content

    def test_marker_in_non_preference_is_literal(self) -> None:
        body = "::instruction::\n" + _AUTO_MINED + "### Q\nhi?\n### A\nhello.\n"
        parsed = _parsed(body)
        section = parsed.sections[0]
        assert section.type == SectionType.INSTRUCTION
        assert section.auto_mined is False
        assert "dlm-auto-mined" in section.content

    def test_bad_score_refused(self) -> None:
        body = (
            "::preference::\n"
            '<!-- dlm-auto-mined: judge_name="sway:preference_judge" '
            'judge_score_chosen="nan" judge_score_rejected="0.1" '
            'mined_at="2026-04-23T20:00:00Z" mined_run_id="7" -->\n'
            "### Prompt\nquestion?\n### Chosen\nbetter\n### Rejected\nworse\n"
        )
        with pytest.raises(FenceError, match="judge scores must be finite"):
            _parsed(body)


class TestAutoMinedMarkerSerialize:
    def test_roundtrip_emits_marker(self) -> None:
        doc = _parsed("::preference::\n### Prompt\nx\n### Chosen\ny\n### Rejected\nz\n")
        section = Section(
            type=SectionType.PREFERENCE,
            content="### Prompt\nquestion?\n### Chosen\nbetter\n### Rejected\nworse\n",
            auto_mined=True,
            judge_name="hf:OpenAssistant/reward-model",
            judge_score_chosen=1.25,
            judge_score_rejected=-0.25,
            mined_at="2026-04-23T20:00:00Z",
            mined_run_id=11,
        )
        updated = type(doc)(
            frontmatter=doc.frontmatter,
            sections=(section,),
            source_path=doc.source_path,
        )
        out = serialize(updated)
        reparsed = parse_text(out)
        mined = reparsed.sections[0]
        assert mined.auto_mined is True
        assert mined.judge_name == "hf:OpenAssistant/reward-model"
        assert mined.mined_run_id == 11

    def test_missing_metadata_refused(self) -> None:
        with pytest.raises(ValueError, match="require metadata fields"):
            Section(
                type=SectionType.PREFERENCE,
                content="### Prompt\nquestion?\n### Chosen\nbetter\n### Rejected\nworse\n",
                auto_mined=True,
                judge_name="sway:preference_judge",
                judge_score_chosen=0.8,
                judge_score_rejected=0.2,
                mined_at="2026-04-23T20:00:00Z",
            )


class TestAutoMinedIdentity:
    def test_section_id_ignores_auto_mined_fields(self) -> None:
        mined = Section(
            type=SectionType.PREFERENCE,
            content="### Prompt\nsame\n### Chosen\na\n### Rejected\nb\n",
            auto_mined=True,
            judge_name="sway:preference_judge",
            judge_score_chosen=0.8,
            judge_score_rejected=0.2,
            mined_at="2026-04-23T20:00:00Z",
            mined_run_id=3,
        )
        authored = Section(
            type=SectionType.PREFERENCE,
            content="### Prompt\nsame\n### Chosen\na\n### Rejected\nb\n",
        )
        assert mined.section_id == authored.section_id
