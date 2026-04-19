"""Grammar coverage for `::preference::` bodies."""

from __future__ import annotations

import pytest

from dlm.data.errors import PreferenceParseError
from dlm.data.preference_parser import PreferenceTriple, parse_preference_body


class TestHappyPath:
    def test_single_triple(self) -> None:
        body = "### Prompt\np\n### Chosen\nc\n### Rejected\nr"
        triples = parse_preference_body(body, section_id="sid")
        assert triples == [PreferenceTriple(prompt="p", chosen="c", rejected="r")]

    def test_multiple_triples_separated_by_blank(self) -> None:
        body = (
            "### Prompt\np1\n### Chosen\nc1\n### Rejected\nr1\n\n"
            "### Prompt\np2\n### Chosen\nc2\n### Rejected\nr2"
        )
        triples = parse_preference_body(body, section_id="sid")
        assert len(triples) == 2
        assert triples[1].prompt == "p2"

    def test_multiline_bodies(self) -> None:
        body = "### Prompt\npart1\npart2\n### Chosen\ncA\ncB\n### Rejected\nrA\nrB"
        triples = parse_preference_body(body, section_id="sid")
        assert triples[0].prompt == "part1\npart2"
        assert triples[0].chosen == "cA\ncB"
        assert triples[0].rejected == "rA\nrB"

    def test_leading_blanks_tolerated(self) -> None:
        body = "\n\n### Prompt\np\n### Chosen\nc\n### Rejected\nr"
        triples = parse_preference_body(body, section_id="sid")
        assert len(triples) == 1


class TestErrors:
    def test_empty_body_raises(self) -> None:
        with pytest.raises(PreferenceParseError, match="no ### Prompt"):
            parse_preference_body("", section_id="sid")

    def test_missing_rejected(self) -> None:
        body = "### Prompt\np\n### Chosen\nc"
        with pytest.raises(PreferenceParseError, match="expected `### Rejected`"):
            parse_preference_body(body, section_id="sid")

    def test_missing_chosen(self) -> None:
        body = "### Prompt\np\n### Rejected\nr"
        with pytest.raises(PreferenceParseError, match="expected `### Chosen`"):
            parse_preference_body(body, section_id="sid")

    def test_wrong_order_rejected(self) -> None:
        body = "### Chosen\nc\n### Prompt\np\n### Rejected\nr"
        with pytest.raises(PreferenceParseError, match="expected `### Prompt`"):
            parse_preference_body(body, section_id="sid")

    def test_empty_prompt_body(self) -> None:
        body = "### Prompt\n### Chosen\nc\n### Rejected\nr"
        with pytest.raises(PreferenceParseError, match="Prompt` body is empty"):
            parse_preference_body(body, section_id="sid")

    def test_empty_chosen_body(self) -> None:
        body = "### Prompt\np\n### Chosen\n### Rejected\nr"
        with pytest.raises(PreferenceParseError, match="Chosen` body is empty"):
            parse_preference_body(body, section_id="sid")

    def test_inline_content_on_header(self) -> None:
        body = "### Prompt inline\np\n### Chosen\nc\n### Rejected\nr"
        with pytest.raises(PreferenceParseError, match="alone on its line"):
            parse_preference_body(body, section_id="sid")

    def test_error_carries_section_id(self) -> None:
        with pytest.raises(PreferenceParseError) as excinfo:
            parse_preference_body("### Prompt\np\n### Chosen\nc", section_id="sid-xyz")
        assert excinfo.value.section_id == "sid-xyz"
        assert excinfo.value.section_line >= 1
