"""Parse `### Q` / `### A` pairs out of an `::instruction::` section body.

Grammar (strict):

    ### Q
    <question body, one or more lines, first blank line ends it>
    ### A
    <answer body, same rule>
    (blank line)
    ### Q
    ...

Rules:

- Headers must be `### Q` / `### A` alone on their line (leading/trailing
  whitespace tolerated). Inline content like `### Q what's this?` is a
  parse error — the body begins on the *next* line.
- Every `### Q` must be followed (after its body) by a matching `### A`.
  An unterminated question, two questions in a row, or a bare `### A`
  raises `InstructionParseError` with the 1-indexed section-relative line
  where the violation was detected.
- Empty question or empty answer bodies are errors — training on an
  empty turn is almost always a mistake.
- Non-header, non-blank lines outside a field body are errors; prose
  that isn't part of a turn belongs in a default PROSE section.
"""

from __future__ import annotations

from dataclasses import dataclass

from dlm.data.errors import InstructionParseError

_Q_HEADER = "### Q"
_A_HEADER = "### A"


@dataclass(frozen=True)
class QAPair:
    """A single instruction turn."""

    question: str
    answer: str


def parse_instruction_body(body: str, *, section_id: str) -> list[QAPair]:
    """Return the list of Q/A pairs in `body`.

    `section_id` is stamped onto any raised `InstructionParseError` so
    the caller can point the user back at the offending `.dlm` section.
    """
    lines = body.splitlines()
    it = _PeekableLines(lines)
    it.skip_blank()

    pairs: list[QAPair] = []
    while not it.eof():
        pairs.append(_parse_pair(it, section_id=section_id))
        it.skip_blank()

    if not pairs:
        raise InstructionParseError(
            "instruction block has no ### Q / ### A pairs",
            section_id=section_id,
            section_line=1,
        )
    return pairs


def _parse_pair(it: _PeekableLines, *, section_id: str) -> QAPair:
    q_line = it.peek_line()
    if not _is_header(q_line, _Q_HEADER):
        raise InstructionParseError(
            f"expected `{_Q_HEADER}` header alone on its line, got {q_line!r}",
            section_id=section_id,
            section_line=it.line_no(),
        )
    it.advance()

    question = _read_field_body(it)
    if not question:
        raise InstructionParseError(
            "### Q body is empty",
            section_id=section_id,
            section_line=it.line_no(),
        )

    a_line = it.peek_line()
    if a_line is None:
        raise InstructionParseError(
            f"### Q without matching `{_A_HEADER}` at end of section",
            section_id=section_id,
            section_line=it.line_no(),
        )
    if not _is_header(a_line, _A_HEADER):
        raise InstructionParseError(
            f"### Q must be followed by `{_A_HEADER}` alone on its line, got {a_line!r}",
            section_id=section_id,
            section_line=it.line_no(),
        )
    it.advance()

    answer = _read_field_body(it)
    if not answer:
        raise InstructionParseError(
            "### A body is empty",
            section_id=section_id,
            section_line=it.line_no(),
        )

    return QAPair(question=question, answer=answer)


def _read_field_body(it: _PeekableLines) -> str:
    """Read until a blank line or the start of another header.

    The terminating blank line is consumed so the outer loop sees the
    next header directly; headers are left for the outer loop.
    """
    buf: list[str] = []
    while not it.eof():
        line = it.peek_line()
        assert line is not None
        if line.strip() == "":
            it.advance()
            break
        if _is_header(line, _Q_HEADER) or _is_header(line, _A_HEADER):
            break
        buf.append(line)
        it.advance()
    return "\n".join(buf).strip()


def _is_header(line: str | None, header: str) -> bool:
    return line is not None and line.strip() == header


class _PeekableLines:
    """Minimal line-at-a-time iterator with 1-indexed line tracking."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self._i = 0

    def peek_line(self) -> str | None:
        if self._i >= len(self._lines):
            return None
        return self._lines[self._i]

    def advance(self) -> None:
        self._i += 1

    def eof(self) -> bool:
        return self._i >= len(self._lines)

    def line_no(self) -> int:
        return self._i + 1

    def skip_blank(self) -> None:
        while not self.eof():
            line = self.peek_line()
            if line is None or line.strip() != "":
                return
            self.advance()
