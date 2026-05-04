"""Grammar coverage for `::instruction::` bodies."""

from __future__ import annotations

import pytest

from dlm.data.errors import InstructionParseError
from dlm.data.instruction_parser import QAPair, parse_instruction_body


class TestHappyPath:
    def test_single_pair(self) -> None:
        body = "### Q\nWhat is 2+2?\n### A\n4"
        pairs = parse_instruction_body(body, section_id="sid1")
        assert pairs == [QAPair(question="What is 2+2?", answer="4")]

    def test_multiple_pairs_separated_by_blank(self) -> None:
        body = "### Q\nQ1?\n### A\nA1\n\n### Q\nQ2?\n### A\nA2\n\n### Q\nQ3?\n### A\nA3"
        pairs = parse_instruction_body(body, section_id="sid1")
        assert len(pairs) == 3
        assert pairs[1] == QAPair(question="Q2?", answer="A2")

    def test_leading_blanks_tolerated(self) -> None:
        body = "\n\n\n### Q\nhi\n### A\nhello"
        pairs = parse_instruction_body(body, section_id="sid1")
        assert len(pairs) == 1

    def test_multiline_question_and_answer(self) -> None:
        body = "### Q\nline 1\nline 2\nline 3\n### A\nansA\nansB"
        pairs = parse_instruction_body(body, section_id="sid1")
        assert pairs[0].question == "line 1\nline 2\nline 3"
        assert pairs[0].answer == "ansA\nansB"

    def test_header_with_trailing_whitespace_tolerated(self) -> None:
        body = "### Q  \nquestion\n### A\t\nanswer"
        pairs = parse_instruction_body(body, section_id="sid1")
        assert len(pairs) == 1


class TestErrors:
    def test_empty_body_raises(self) -> None:
        with pytest.raises(InstructionParseError, match="no ### Q / ### A pairs"):
            parse_instruction_body("", section_id="sid")

    def test_only_whitespace_raises(self) -> None:
        with pytest.raises(InstructionParseError, match="no ### Q"):
            parse_instruction_body("\n\n  \n", section_id="sid")

    def test_unterminated_question_raises(self) -> None:
        with pytest.raises(InstructionParseError, match="without matching"):
            parse_instruction_body("### Q\nquestion", section_id="sid")

    def test_q_without_a_before_next_q(self) -> None:
        body = "### Q\nq1\n### Q\nq2\n### A\na2"
        with pytest.raises(InstructionParseError, match="must be followed by"):
            parse_instruction_body(body, section_id="sid")

    def test_bare_a_without_q(self) -> None:
        with pytest.raises(InstructionParseError, match="expected `### Q` header"):
            parse_instruction_body("### A\ndangling", section_id="sid")

    def test_inline_content_on_header(self) -> None:
        with pytest.raises(InstructionParseError, match="alone on its line"):
            parse_instruction_body("### Q What's this?\n### A\nfoo", section_id="sid")

    def test_empty_question_body(self) -> None:
        with pytest.raises(InstructionParseError, match="Q body is empty"):
            parse_instruction_body("### Q\n### A\nfoo", section_id="sid")

    def test_empty_answer_body(self) -> None:
        with pytest.raises(InstructionParseError, match="A body is empty"):
            parse_instruction_body("### Q\nq\n### A\n", section_id="sid")

    def test_error_carries_section_id(self) -> None:
        with pytest.raises(InstructionParseError) as excinfo:
            parse_instruction_body("### Q\nq", section_id="sid-abc")
        assert excinfo.value.section_id == "sid-abc"
        assert excinfo.value.section_line >= 1

    def test_prose_between_pairs_absorbed_into_prior_answer(self) -> None:
        """Prose between Q/A pairs is folded into the previous answer.

        Bodies terminate at the next ``### Q`` / ``### A`` header (or
        EOF), so a paragraph wedged between two pairs becomes part of
        the first pair's answer rather than raising. Authors should
        either format the stray prose as a separate PROSE section
        outside the ``::instruction::`` block, or accept it gets
        appended to the prior answer.
        """
        body = "### Q\nq1\n### A\na1\n\nsome prose not in a section\n### Q\nq2\n### A\na2"
        pairs = parse_instruction_body(body, section_id="sid")
        assert len(pairs) == 2
        assert pairs[0].answer == "a1\n\nsome prose not in a section"
        assert pairs[1] == QAPair(question="q2", answer="a2")


class TestBlankLinePreservation:
    """Bodies preserve blank lines so multi-paragraph answers and fenced
    code blocks (which routinely separate imports from the call site
    with a blank) parse correctly."""

    def test_answer_with_blank_line_inside_fenced_code_block(self) -> None:
        body = (
            "### Q\nHow do I append to a log file?\n"
            "### A\n"
            "Use ``logger_type%add_log_file`` with ``position='append'``:\n"
            "```fortran\n"
            "use stdlib_logger, only: logger_type\n"
            "type(logger_type) :: log\n"
            "integer :: stat\n"
            "\n"
            "call log%add_log_file('app.log', stat, position='append')\n"
            "call log%log_information('app started')\n"
            "```\n"
            "Default level is ``information``; raise via ``configuration``."
        )
        pairs = parse_instruction_body(body, section_id="sid")
        assert len(pairs) == 1
        assert "call log%add_log_file" in pairs[0].answer
        # Blank line preserved inside the fence.
        assert "\n\ncall log%add_log_file" in pairs[0].answer

    def test_multi_paragraph_answer(self) -> None:
        body = (
            "### Q\nWhat is X?\n"
            "### A\n"
            "Paragraph one explains the basics.\n"
            "\n"
            "Paragraph two adds context that wouldn't fit in one breath.\n"
            "\n"
            "Paragraph three lands the recommendation."
        )
        pairs = parse_instruction_body(body, section_id="sid")
        assert pairs[0].answer.count("\n\n") == 2

    def test_two_pairs_with_multi_paragraph_answers(self) -> None:
        body = "### Q\nq1?\n### A\nfirst para\n\nsecond para\n### Q\nq2?\n### A\nans2"
        pairs = parse_instruction_body(body, section_id="sid")
        assert len(pairs) == 2
        assert pairs[0].answer == "first para\n\nsecond para"
        assert pairs[1].answer == "ans2"

    def test_trailing_blank_lines_stripped(self) -> None:
        body = "### Q\nq?\n### A\nthe answer\n\n\n"
        pairs = parse_instruction_body(body, section_id="sid")
        assert pairs[0].answer == "the answer"
