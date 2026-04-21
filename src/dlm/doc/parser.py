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

from dlm.doc.errors import (
    FenceError,
    FrontmatterError,
)
from dlm.doc.schema import DlmFrontmatter
from dlm.doc.sections import Section, SectionType
from dlm.doc.versioned import validate_versioned
from dlm.io.text import read_text

# Schema v7 marker: sections written back by `dlm harvest` carry a
# magic-comment first line inside the fenced section body. The parser
# recognizes it and moves the metadata to `Section.auto_harvest` +
# `Section.harvest_source`; it is not user-authored content.
_HARVEST_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r'^<!-- dlm-auto-harvest: source="([^"]*)" -->$'
)

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
    sections = _tokenize_body(body, body_start_line=body_start_line, path=path)
    return ParsedDlm(
        frontmatter=frontmatter,
        sections=tuple(sections),
        source_path=path,
    )


# --- internals ----------------------------------------------------------------


_FRONTMATTER_DELIM: Final = "---"

# A fence line is one of:
#   `::<type>::`                      — bare fence
#   `::<type>#<adapter>::`            — adapter-routed fence
#   `::<type> key="val" key="val"::`  — attribute fence (IMAGE, schema v10+)
#
# - `<type>` is one of `SectionType` (validated in `_resolve_fence_type`).
# - `<adapter>` matches the schema's adapter-name grammar: lowercase
#   alpha start + `[a-z0-9_]` tail, ≤32 chars. Keeps store paths safe
#   and log output readable.
# - Attribute form values are double-quoted, ASCII-only, and cannot
#   contain newlines (enforced at parse time). Currently only the
#   IMAGE type uses attributes (`path`, `alt`); adding more is a
#   compatible extension.
_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"^::([A-Za-z0-9_#-]+)::$")
_ATTR_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r'^::([a-z][a-z0-9_]*)((?:\s+[a-z][a-z0-9_]*="[^"\n]*")+)\s*::$'
)
_ATTR_KV_RE: Final[re.Pattern[str]] = re.compile(r'([a-z][a-z0-9_]*)="([^"\n]*)"')
_ADAPTER_SUFFIX_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_CODE_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"^```")

# Per-type attribute grammar. Keys marked required must appear; unknown
# keys raise a FenceError. Expanded by future multi-modal types.
_FENCE_ATTR_SPEC: Final[dict[str, tuple[frozenset[str], frozenset[str]]]] = {
    # IMAGE: `path` required, `alt` optional.
    "image": (frozenset({"path"}), frozenset({"path", "alt"})),
}


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
    """YAML-parse → migrate-if-needed → Pydantic-validate.

    The Sprint 12b migration dispatcher runs between YAML parse and
    Pydantic validate, so an older-but-known schema is upgraded to the
    current shape before `extra="forbid"` enforcement.
    """
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

    return validate_versioned(raw, path=path)


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


def _tokenize_body(body: str, *, body_start_line: int, path: Path | None) -> list[Section]:
    """Split body into Section list.

    Rules:

    - Active type starts as PROSE.
    - A bare fence line is exactly `^::<type>::$` or `^::<type>#<adapter>::$`.
    - An attribute fence line is `^::<type> key="val" ...::$` — currently
      only IMAGE uses this form.
    - Unknown fences raise `FenceError`.
    - Triple-backtick code blocks (```...```) suppress fence
      interpretation for their contents.
    - Empty PROSE sections (between two fences back-to-back) are elided.
    """
    # The canonical layout emits a single blank line between the closing
    # `---` and the first body line. Strip one leading LF so section
    # content doesn't accumulate that separator line on every round-trip.
    if body.startswith("\n"):
        body = body[1:]
        body_start_line += 1

    # Likewise, files canonically end with a trailing LF; `split("\n")`
    # would otherwise produce a spurious empty trailing element.
    if body.endswith("\n"):
        body = body[:-1]

    lines = body.split("\n") if body else []
    sections: list[Section] = []
    in_code_block = False
    current_type = SectionType.PROSE
    current_adapter: str | None = None
    current_media_path: str | None = None
    current_media_alt: str | None = None
    current_lines: list[str] = []
    current_start_line = body_start_line

    def flush() -> None:
        # Schema v7: non-PROSE sections may carry a harvest marker as
        # the first body line. Lift it into Section fields before the
        # content hash sees it, so a harvested section's `section_id`
        # matches a hand-authored section with identical content —
        # provenance is metadata, not identity.
        lines_for_content = list(current_lines)
        auto_harvest = False
        harvest_source: str | None = None
        if current_type not in (SectionType.PROSE, SectionType.IMAGE) and lines_for_content:
            marker_match = _HARVEST_MARKER_RE.match(lines_for_content[0])
            if marker_match:
                auto_harvest = True
                harvest_source = marker_match.group(1)
                lines_for_content = lines_for_content[1:]
        content = "\n".join(lines_for_content)
        # Elide empty PROSE sections (no content at all).
        if current_type == SectionType.PROSE and not content.strip() and not current_lines:
            return
        if current_type == SectionType.PROSE and not content.strip():
            # Purely-whitespace prose between fences: drop, keeps round-trip tidy.
            return
        sections.append(
            Section(
                type=current_type,
                content=content,
                start_line=current_start_line,
                adapter=current_adapter,
                auto_harvest=auto_harvest,
                harvest_source=harvest_source,
                media_path=current_media_path,
                media_alt=current_media_alt,
            ),
        )

    for idx, line in enumerate(lines):
        source_line = body_start_line + idx

        # Track fenced code blocks to suppress fence interpretation.
        if _CODE_FENCE_RE.match(line):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        if not in_code_block:
            attr_match = _ATTR_FENCE_RE.match(line)
            if attr_match:
                fence_type, attrs = _resolve_attr_fence(attr_match, source_line, path)
                flush()
                current_type = fence_type
                current_adapter = None
                current_media_path = attrs.get("path")
                current_media_alt = attrs.get("alt")
                current_lines = []
                current_start_line = source_line
                continue
            match = _FENCE_RE.match(line)
            if match:
                fence_name = match.group(1)
                fence_type, fence_adapter = _resolve_fence_type(fence_name, source_line, path)
                if fence_type in _FENCE_ATTR_SPEC:
                    raise FenceError(
                        f"fence '::{fence_name}::' requires attributes "
                        f"(expected e.g. `::{fence_type.value} path=\"...\"::`)",
                        path=path,
                        line=source_line,
                        col=1,
                    )
                flush()
                current_type = fence_type
                current_adapter = fence_adapter
                current_media_path = None
                current_media_alt = None
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


def _resolve_attr_fence(
    match: re.Match[str], line: int, path: Path | None
) -> tuple[SectionType, dict[str, str]]:
    """Validate an attribute-form fence and return (type, attrs).

    The attribute grammar is type-specific — `_FENCE_ATTR_SPEC` names
    the required and allowed keys per type. Required-but-missing keys
    and unknown keys raise `FenceError`; duplicate keys raise too so
    `path="a" path="b"` can't silently pick one.
    """
    fence_name = match.group(1)
    attr_blob = match.group(2)
    try:
        section_type = SectionType(fence_name)
    except ValueError as exc:
        raise FenceError(
            f"unknown attribute fence '::{fence_name}...::'; attribute form "
            f"is only valid for types {sorted(_FENCE_ATTR_SPEC)}",
            path=path,
            line=line,
            col=1,
        ) from exc
    if fence_name not in _FENCE_ATTR_SPEC:
        raise FenceError(
            f"fence '::{fence_name}...::' does not take attributes",
            path=path,
            line=line,
            col=1,
        )
    required, allowed = _FENCE_ATTR_SPEC[fence_name]

    attrs: dict[str, str] = {}
    for kv in _ATTR_KV_RE.finditer(attr_blob):
        key = kv.group(1)
        value = kv.group(2)
        if key in attrs:
            raise FenceError(
                f"fence '::{fence_name}...::' repeats attribute {key!r}",
                path=path,
                line=line,
                col=1,
            )
        if key not in allowed:
            raise FenceError(
                f"fence '::{fence_name}...::' has unknown attribute {key!r} "
                f"(allowed: {sorted(allowed)})",
                path=path,
                line=line,
                col=1,
            )
        if not value.isascii():
            raise FenceError(
                f"fence '::{fence_name}...::' attribute {key!r} contains non-ASCII characters",
                path=path,
                line=line,
                col=1,
            )
        attrs[key] = value

    missing = required - attrs.keys()
    if missing:
        raise FenceError(
            f"fence '::{fence_name}...::' is missing required attribute(s) {sorted(missing)}",
            path=path,
            line=line,
            col=1,
        )
    return section_type, attrs


def _resolve_fence_type(name: str, line: int, path: Path | None) -> tuple[SectionType, str | None]:
    """Map a fence name to `(SectionType, adapter_name|None)` or raise.

    Multi-adapter fences carry a `#<adapter>` suffix; the adapter part is
    validated against the same grammar the schema uses. A fence like
    `::instruction#::` (trailing hash but no name) or `::foo#bar::` (bad
    base) raises `FenceError`.
    """
    if "#" in name:
        base, _, adapter = name.partition("#")
        if not adapter:
            raise FenceError(
                f"fence '::{name}::' has an empty adapter suffix after '#'",
                path=path,
                line=line,
                col=1,
            )
        if not _ADAPTER_SUFFIX_RE.fullmatch(adapter):
            raise FenceError(
                f"fence '::{name}::' has an invalid adapter name "
                f"{adapter!r} (must match {_ADAPTER_SUFFIX_RE.pattern})",
                path=path,
                line=line,
                col=1,
            )
    else:
        base, adapter = name, None

    try:
        section_type = SectionType(base)
    except ValueError as exc:
        raise FenceError(
            f"unknown section fence '::{name}::'; valid types are {[t.value for t in SectionType]}",
            path=path,
            line=line,
            col=1,
        ) from exc
    return section_type, adapter or None
