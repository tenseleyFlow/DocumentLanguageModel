"""Parse `### Prompt` / `### Chosen` / `### Rejected` triples from a
`::preference::` section body.

Grammar (strict):

    ### Prompt
    <prompt body>
    ### Chosen
    <chosen body>
    ### Rejected
    <rejected body>
    (blank line)
    ### Prompt
    ...

The three headers must appear in order (Prompt → Chosen → Rejected) for
each triple. Missing, duplicated, or reordered headers raise
`PreferenceParseError`. Empty field bodies are errors — DPO on empty
text is never intentional.

Sprint 07 only parses + validates. The DPO consumer is Sprint 17.
"""

from __future__ import annotations

from dataclasses import dataclass

from dlm.data.errors import PreferenceParseError

_PROMPT = "### Prompt"
_CHOSEN = "### Chosen"
_REJECTED = "### Rejected"
_ALL_HEADERS = (_PROMPT, _CHOSEN, _REJECTED)


@dataclass(frozen=True)
class PreferenceTriple:
    """A single preference example: prompt + chosen + rejected completion."""

    prompt: str
    chosen: str
    rejected: str


def parse_preference_body(body: str, *, section_id: str) -> list[PreferenceTriple]:
    """Return the list of preference triples in `body`."""
    lines = body.splitlines()
    it = _PeekableLines(lines)
    it.skip_blank()

    triples: list[PreferenceTriple] = []
    while not it.eof():
        triples.append(_parse_triple(it, section_id=section_id))
        it.skip_blank()

    if not triples:
        raise PreferenceParseError(
            "preference block has no ### Prompt / ### Chosen / ### Rejected triples",
            section_id=section_id,
            section_line=1,
        )
    return triples


def _parse_triple(it: _PeekableLines, *, section_id: str) -> PreferenceTriple:
    prompt = _parse_field(it, expected=_PROMPT, section_id=section_id)
    chosen = _parse_field(it, expected=_CHOSEN, section_id=section_id)
    rejected = _parse_field(it, expected=_REJECTED, section_id=section_id)
    return PreferenceTriple(prompt=prompt, chosen=chosen, rejected=rejected)


def _parse_field(it: _PeekableLines, *, expected: str, section_id: str) -> str:
    line = it.peek_line()
    if line is None:
        raise PreferenceParseError(
            f"expected `{expected}` header, got end of section",
            section_id=section_id,
            section_line=it.line_no(),
        )
    if line.strip() != expected:
        raise PreferenceParseError(
            f"expected `{expected}` header alone on its line, got {line!r}",
            section_id=section_id,
            section_line=it.line_no(),
        )
    it.advance()

    body = _read_field_body(it)
    if not body:
        raise PreferenceParseError(
            f"`{expected}` body is empty",
            section_id=section_id,
            section_line=it.line_no(),
        )
    return body


def _read_field_body(it: _PeekableLines) -> str:
    """Read until a blank line or the next recognized header."""
    buf: list[str] = []
    while not it.eof():
        line = it.peek_line()
        assert line is not None
        if line.strip() == "":
            it.advance()
            break
        if line.strip() in _ALL_HEADERS:
            break
        buf.append(line)
        it.advance()
    return "\n".join(buf).strip()


class _PeekableLines:
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
