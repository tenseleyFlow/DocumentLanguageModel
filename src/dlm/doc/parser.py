"""Parse `.dlm` files into validated `ParsedDlm` values.

Flow:

    read bytes → dlm.io.text.read_text (UTF-8 strict, BOM strip, CRLF→LF)
              → split frontmatter and body on the two `---` delimiters
              → YAML-parse the frontmatter
              → Pydantic validate → DlmFrontmatter
              → check dlm_version (sprint 12b owns migration)
              → tokenize body into Section list (code-fence aware)
              → return ParsedDlm(frozen)

Errors are always typed (`DlmParseError` subclasses) and carry
`path:line:col` location info for the CLI reporter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml
from pydantic import ValidationError

from dlm.doc.errors import (
    DlmVersionError,
    FenceError,
    FrontmatterError,
    SchemaValidationError,
)
from dlm.doc.schema import CURRENT_SCHEMA_VERSION, DlmFrontmatter
from dlm.doc.sections import Section, SectionType
from dlm.io.text import read_text

# --- public surface -----------------------------------------------------------


@dataclass(frozen=True)
class ParsedDlm:
    """Immutable result of parsing a `.dlm` document."""

    frontmatter: DlmFrontmatter
    sections: tuple[Section, ...]
    source_path: Path | None = None


def parse_file(path: Path) -> ParsedDlm:
    """Read `path` as UTF-8 and parse it."""
    text = read_text(path)
    return parse_text(text, path=path)


def parse_text(text: str, *, path: Path | None = None) -> ParsedDlm:
    """Parse the already-decoded text body of a `.dlm` document.

    Assumes the caller has applied UTF-8 decoding + LF normalization (the
    `dlm.io.text.read_text` helper does this automatically).
    """
    frontmatter_text, body, body_start_line = _split_frontmatter(text, path=path)
    frontmatter = _validate_frontmatter(frontmatter_text, path=path)
    _check_version(frontmatter.dlm_version, path=path)
    sections = _tokenize_body(body, body_start_line=body_start_line, path=path)
    return ParsedDlm(
        frontmatter=frontmatter,
        sections=tuple(sections),
        source_path=path,
    )


# --- internals ----------------------------------------------------------------


_FRONTMATTER_DELIM: Final = "---"
_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"^::([A-Za-z0-9_#-]+)::$")
_CODE_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"^```")


def _split_frontmatter(text: str, *, path: Path | None) -> tuple[str, str, int]:
    """Return (frontmatter_yaml, body_text, body_start_line_1indexed).

    The first line MUST be `---`. The next `---` line closes the block.
    Everything after is body. Missing either delimiter is an error.
    """
    lines = text.split("\n")
    if not lines or lines[0] != _FRONTMATTER_DELIM:
        observed = repr(lines[0]) if lines else "''"
        raise FrontmatterError(
            f"expected '---' on line 1 to open frontmatter, got {observed}",
            path=path,
            line=1,
            col=1,
        )
    # Find the closing delimiter.
    for i in range(1, len(lines)):
        if lines[i] == _FRONTMATTER_DELIM:
            yaml_text = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :])
            # body starts on the line after the closer (1-indexed).
            return yaml_text, body, i + 2
    raise FrontmatterError(
        "no closing '---' found for frontmatter block",
        path=path,
        line=1,
        col=1,
    )


def _validate_frontmatter(yaml_text: str, *, path: Path | None) -> DlmFrontmatter:
    """YAML-parse and Pydantic-validate the frontmatter text."""
    try:
        raw = yaml.safe_load(yaml_text) if yaml_text.strip() else {}
    except yaml.YAMLError as exc:
        line, col = _yaml_error_location(exc)
        raise FrontmatterError(f"invalid YAML: {exc}", path=path, line=line, col=col) from exc

    if not isinstance(raw, dict):
        raise FrontmatterError(
            f"frontmatter must be a mapping, got {type(raw).__name__}",
            path=path,
            line=2,
        )

    try:
        return DlmFrontmatter.model_validate(raw)
    except ValidationError as exc:
        # Pydantic doesn't know source-file line numbers; we cite the
        # start of the frontmatter block and include the full field path
        # in the message.
        raise SchemaValidationError(
            _format_pydantic_error(exc),
            path=path,
            line=2,
        ) from exc


def _yaml_error_location(exc: yaml.YAMLError) -> tuple[int, int]:
    """Extract 1-indexed (line, col) from a YAMLError, if present.

    The exception's `problem_mark` attribute is 0-indexed internally.
    """
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    if mark is None:
        return 0, 0
    # The frontmatter parser feeds yaml the content *without* its
    # delimiters, so the reported line is offset by 1 (the opening `---`).
    return mark.line + 2, mark.column + 1


def _format_pydantic_error(exc: ValidationError) -> str:
    """Human-readable single-line summary of a Pydantic error."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        msg = err.get("msg", "invalid value")
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts) or "validation failed"


def _check_version(version: int, *, path: Path | None) -> None:
    if version == CURRENT_SCHEMA_VERSION:
        return
    if version > CURRENT_SCHEMA_VERSION:
        raise DlmVersionError(
            f"dlm_version {version} is newer than this parser ({CURRENT_SCHEMA_VERSION}); "
            "upgrade dlm or check the source's schema",
            path=path,
            line=2,
        )
    # Older-but-positive versions route to the migration framework
    # (Sprint 12b). Until that sprint lands, we refuse with a pointer.
    raise DlmVersionError(
        f"dlm_version {version} requires migration to {CURRENT_SCHEMA_VERSION}; "
        "run `dlm migrate <path>` (Sprint 12b) to upgrade the document",
        path=path,
        line=2,
    )


def _tokenize_body(body: str, *, body_start_line: int, path: Path | None) -> list[Section]:
    """Split body into Section list.

    Rules:

    - Active type starts as PROSE.
    - A fence line is exactly `^::<type>::$` with no surrounding
      whitespace. Recognized types are the values of `SectionType`.
    - Unknown fences raise `FenceError`.
    - Triple-backtick code blocks (```...```) suppress fence
      interpretation for their contents.
    - Empty PROSE sections (between two fences back-to-back) are elided.
    """
    lines = body.split("\n") if body else []
    sections: list[Section] = []
    in_code_block = False
    current_type = SectionType.PROSE
    current_lines: list[str] = []
    current_start_line = body_start_line

    def flush() -> None:
        content = "\n".join(current_lines)
        # Elide empty PROSE sections (no content at all).
        if current_type == SectionType.PROSE and not content.strip() and not current_lines:
            return
        if current_type == SectionType.PROSE and not content.strip():
            # Purely-whitespace prose between fences: drop, keeps round-trip tidy.
            return
        sections.append(
            Section(type=current_type, content=content, start_line=current_start_line),
        )

    for idx, line in enumerate(lines):
        source_line = body_start_line + idx

        # Track fenced code blocks to suppress fence interpretation.
        if _CODE_FENCE_RE.match(line):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        if not in_code_block:
            match = _FENCE_RE.match(line)
            if match:
                fence_name = match.group(1)
                fence_type = _resolve_fence_type(fence_name, source_line, path)
                flush()
                current_type = fence_type
                current_lines = []
                current_start_line = source_line
                continue

        current_lines.append(line)

    if in_code_block:
        raise FenceError(
            "unterminated triple-backtick code block in body",
            path=path,
            line=current_start_line,
        )

    flush()
    return sections


def _resolve_fence_type(name: str, line: int, path: Path | None) -> SectionType:
    """Map a fence name to a known SectionType or raise."""
    # Multi-adapter fences like `instruction#tone` are Phase 4; accept the
    # base type but record the adapter routing via the annotation (Sprint 20
    # consumes). For v1 we only accept bare names.
    base = name.split("#", 1)[0]
    try:
        return SectionType(base)
    except ValueError as exc:
        raise FenceError(
            f"unknown section fence '::{name}::'; valid types are {[t.value for t in SectionType]}",
            path=path,
            line=line,
            col=1,
        ) from exc
