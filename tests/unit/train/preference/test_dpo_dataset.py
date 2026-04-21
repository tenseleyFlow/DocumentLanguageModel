"""DPO dataset assembly — preference sections → DPOTrainer rows."""

from __future__ import annotations

import pytest

from dlm.doc.sections import Section, SectionType
from dlm.train.preference.dpo_dataset import (
    build_dpo_dataset,
    extract_preference_triples,
    preference_rows,
)

_PREF_BODY_ONE = (
    "### Prompt\nWhat time is it?\n### Chosen\nIt is 3 PM.\n### Rejected\nTime is an illusion.\n"
)

_PREF_BODY_TWO = (
    "### Prompt\nWho are you?\n"
    "### Chosen\nI am an assistant.\n"
    "### Rejected\nI am nobody.\n"
    "\n"
    "### Prompt\nHow many moons?\n"
    "### Chosen\nEarth has one.\n"
    "### Rejected\nSeventeen.\n"
)


def _pref(body: str) -> Section:
    return Section(type=SectionType.PREFERENCE, content=body, start_line=1)


def _prose(body: str) -> Section:
    return Section(type=SectionType.PROSE, content=body, start_line=1)


class TestExtractPreferenceTriples:
    def test_one_triple_one_section(self) -> None:
        pairs = extract_preference_triples([_pref(_PREF_BODY_ONE)])
        assert len(pairs) == 1
        section, triple = pairs[0]
        assert section.type is SectionType.PREFERENCE
        assert triple.prompt == "What time is it?"
        assert triple.chosen == "It is 3 PM."
        assert triple.rejected == "Time is an illusion."

    def test_multiple_triples_in_one_section(self) -> None:
        pairs = extract_preference_triples([_pref(_PREF_BODY_TWO)])
        assert len(pairs) == 2
        assert pairs[0][1].prompt == "Who are you?"
        assert pairs[1][1].prompt == "How many moons?"
        # same owning section, different triples
        assert pairs[0][0] is pairs[1][0]

    def test_skips_non_preference_sections(self) -> None:
        sections = [
            _prose("just a paragraph"),
            _pref(_PREF_BODY_ONE),
            _prose("another paragraph"),
        ]
        pairs = extract_preference_triples(sections)
        assert len(pairs) == 1

    def test_empty_input_returns_empty(self) -> None:
        assert extract_preference_triples([]) == []

    def test_no_preference_sections_returns_empty(self) -> None:
        assert extract_preference_triples([_prose("hi")]) == []


class TestPreferenceRows:
    def test_row_shape(self) -> None:
        rows = preference_rows([_pref(_PREF_BODY_ONE)])
        assert rows == [
            {
                "prompt": "What time is it?",
                "chosen": "It is 3 PM.",
                "rejected": "Time is an illusion.",
                "_dlm_section_id": _pref(_PREF_BODY_ONE).section_id,
                "_dlm_sub_index": 0,
            }
        ]

    def test_sub_index_increments_within_section(self) -> None:
        rows = preference_rows([_pref(_PREF_BODY_TWO)])
        assert len(rows) == 2
        assert rows[0]["_dlm_sub_index"] == 0
        assert rows[1]["_dlm_sub_index"] == 1
        assert rows[0]["_dlm_section_id"] == rows[1]["_dlm_section_id"]

    def test_sub_index_restarts_per_section(self) -> None:
        rows = preference_rows([_pref(_PREF_BODY_ONE), _pref(_PREF_BODY_TWO)])
        # two different section_ids → each restarts at 0
        by_section: dict[str, list[int]] = {}
        for row in rows:
            by_section.setdefault(row["_dlm_section_id"], []).append(row["_dlm_sub_index"])
        assert all(indices[0] == 0 for indices in by_section.values())

    def test_empty_document_yields_no_rows(self) -> None:
        assert preference_rows([]) == []


class TestBuildDpoDataset:
    def test_returns_hf_dataset_with_correct_columns(self) -> None:
        ds = build_dpo_dataset([_pref(_PREF_BODY_TWO)])
        assert len(ds) == 2
        assert set(ds.column_names) == {
            "prompt",
            "chosen",
            "rejected",
            "_dlm_section_id",
            "_dlm_sub_index",
        }
        row0 = ds[0]
        assert row0["prompt"] == "Who are you?"
        assert row0["chosen"] == "I am an assistant."
        assert row0["rejected"] == "I am nobody."

    def test_empty_dataset_has_zero_rows(self) -> None:
        ds = build_dpo_dataset([_prose("no prefs here")])
        assert len(ds) == 0


class TestParserErrorPropagates:
    def test_malformed_preference_body_raises(self) -> None:
        # Missing the Rejected header → PreferenceParseError bubbles up.
        from dlm.data.errors import PreferenceParseError

        broken = "### Prompt\nhi\n### Chosen\nhello\n"
        with pytest.raises(PreferenceParseError):
            extract_preference_triples([_pref(broken)])
