"""Serialize a `ParsedDlm` back to canonical `.dlm` text.

Contract:

- `serialize(parse_text(t))` may differ from `t` (whitespace/quoting
  normalization), but applying the pipeline a second time is a no-op:
  `serialize(parse_text(serialize(parse_text(t)))) == serialize(parse_text(t))`.
- Frontmatter key order is deterministic (see `_FRONTMATTER_ORDER`).
- Nested mappings (`training`, `export`) preserve the schema's declared
  field order.
- Section content is emitted verbatim; fence lines are regenerated.
- Output uses LF line endings and ends with a single trailing newline.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from pydantic import BaseModel

from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import DlmFrontmatter, ExportConfig, TrainingConfig
from dlm.doc.sections import Section, SectionType

# Top-level frontmatter key order.
_FRONTMATTER_ORDER: Final[tuple[str, ...]] = (
    "dlm_id",
    "dlm_version",
    "base_model",
    "training",
    "export",
    "system_prompt",
)


def serialize(parsed: ParsedDlm) -> str:
    """Produce canonical `.dlm` text for `parsed`.

    Always ends with `\\n`.
    """
    parts: list[str] = [_serialize_frontmatter(parsed.frontmatter), "\n"]
    for i, section in enumerate(parsed.sections):
        if i > 0:
            parts.append("\n")
        parts.append(_serialize_section(section))
    rendered = "".join(parts)
    if not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


# --- frontmatter --------------------------------------------------------------


def _serialize_frontmatter(fm: DlmFrontmatter) -> str:
    lines: list[str] = ["---"]
    for key in _FRONTMATTER_ORDER:
        value = getattr(fm, key, None)
        if key == "system_prompt":
            if value is None:
                continue
            lines.extend(_emit_block_scalar(key, value))
            continue
        if isinstance(value, TrainingConfig | ExportConfig):
            nested = _emit_nested_mapping(value, indent=2)
            if not nested:
                # All-default nested block — skip the header too so we
                # don't emit an empty `training:` line (audit-05 M2).
                continue
            lines.append(f"{key}:")
            lines.extend(nested)
            continue
        lines.append(f"{key}: {_scalar(value)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _emit_nested_mapping(model: BaseModel, *, indent: int) -> list[str]:
    """Emit a nested training/export/dpo block.

    Audit-05 M2: suppress fields that equal their schema default so
    re-serializing a minimal `.dlm` doesn't bloat it with every
    inlined default. Idempotency (Sprint 03 DoD) is preserved — the
    parser's defaults match the suppressed values, so round-trip
    stability holds at the model level.

    Nested `BaseModel` values (e.g. `TrainingConfig.dpo` from Sprint 17)
    recurse with deeper indent; all-default sub-blocks are skipped.
    """
    pad = " " * indent
    lines: list[str] = []
    # model_fields preserves declaration order. Required fields (no
    # default / default_factory) must always emit; optional fields are
    # suppressed when they equal their schema default. Constructing
    # `model.__class__()` would fail for models with required fields
    # (e.g. SourceDirective.path).
    from pydantic_core import PydanticUndefined

    for field_name, field_info in model.__class__.model_fields.items():
        value = getattr(model, field_name)
        if field_info.default is not PydanticUndefined and value == field_info.default:
            continue
        if (
            field_info.default is PydanticUndefined
            and field_info.default_factory is not None
            and value == field_info.default_factory()  # type: ignore[call-arg]
        ):
            continue
        if isinstance(value, BaseModel):
            nested = _emit_nested_mapping(value, indent=indent + 2)
            if not nested:
                continue
            lines.append(f"{pad}{field_name}:")
            lines.extend(nested)
            continue
        if (
            isinstance(value, dict)
            and value
            and all(isinstance(v, BaseModel) for v in value.values())
        ):
            # `dict[str, BaseModel]` (e.g. training.adapters) — emit
            # each entry as a nested mapping. The key is the dict
            # key; the value is the BaseModel's non-default fields.
            lines.append(f"{pad}{field_name}:")
            for k, v in value.items():
                lines.append(f"{pad}  {k}:")
                nested = _emit_nested_mapping(v, indent=indent + 4)
                if nested:
                    lines.extend(nested)
                else:
                    # All-default AdapterConfig: emit explicit `{}` so
                    # YAML has a valid mapping value rather than bare key.
                    lines[-1] = f"{pad}  {k}: {{}}"
            continue
        if (
            isinstance(value, list | tuple)
            and value
            and all(isinstance(v, BaseModel) for v in value)
        ):
            # `tuple[BaseModel, ...]` / `list[BaseModel]` (e.g.
            # training.sources). YAML list of nested mappings — each
            # entry's first field emits with the `-` marker, subsequent
            # fields indent aligned.
            lines.append(f"{pad}{field_name}:")
            for item in value:
                nested = _emit_nested_mapping(item, indent=indent + 4)
                if not nested:
                    lines.append(f"{pad}  - {{}}")
                    continue
                # Replace the first-field indent with `  - ` to start
                # the list item; keep the rest at `indent + 4`.
                first = nested[0]
                prefix = f"{pad}  - "
                lines.append(prefix + first[len(pad) + 4 :])
                lines.extend(nested[1:])
            continue
        lines.append(f"{pad}{field_name}: {_scalar(value)}")
    return lines


def _emit_block_scalar(key: str, value: str) -> list[str]:
    """YAML `|` block scalar: preserves line breaks verbatim."""
    lines: list[str] = [f"{key}: |"]
    for line in value.splitlines():
        lines.append(f"  {line}")
    return lines


def _scalar(value: object) -> str:
    """Render a scalar value in YAML-compatible form.

    Conservative quoting: quote strings that could be misparsed (contain
    whitespace, `:`, `#`, or look like a reserved scalar).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return _format_number(value)
    if isinstance(value, str):
        return _format_string(value)
    if isinstance(value, list | tuple):
        return _format_list(value)
    if value is None:
        return "null"
    return str(value)


def _format_number(value: float | int) -> str:
    """Render a numeric YAML scalar.

    Integers serialize via `str()`; floats via `repr()` so `2e-4` round-trips
    to `0.0002` cleanly. The `_scalar` dispatcher routes bools away before
    we get here, so no bool guard is needed.
    """
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0.0"
    return repr(value)


def _format_string(value: str) -> str:
    if not value:
        return '""'
    if _needs_quoting(value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


_RESERVED_UNQUOTED = frozenset(
    {
        "true",
        "false",
        "null",
        "yes",
        "no",
        "on",
        "off",
        "~",
    }
)


def _needs_quoting(value: str) -> bool:
    if value.lower() in _RESERVED_UNQUOTED:
        return True
    if any(ch in value for ch in " \t\n#\"':&*!|>?%@`{}[]"):
        return True
    # Leading `-` or `,` would be parsed as a YAML list element.
    return value.startswith(("-", ","))


def _format_list(items: Iterable[object]) -> str:
    """Inline flow-style list: `[a, b, c]`."""
    rendered = [_scalar(item) for item in items]
    return "[" + ", ".join(rendered) + "]"


# --- sections -----------------------------------------------------------------


def _serialize_section(section: Section) -> str:
    if section.type == SectionType.PROSE:
        body = section.content
        if not body.endswith("\n"):
            body += "\n"
        return body
    if section.type == SectionType.IMAGE:
        attrs: list[str] = []
        if section.media_path is not None:
            attrs.append(f'path="{section.media_path}"')
        if section.media_alt is not None:
            attrs.append(f'alt="{section.media_alt}"')
        attr_blob = (" " + " ".join(attrs)) if attrs else ""
        fence = f"::{section.type.value}{attr_blob}::\n"
        body = section.content
        if body and not body.endswith("\n"):
            body += "\n"
        return fence + body
    if section.type == SectionType.AUDIO:
        attrs = []
        if section.media_path is not None:
            attrs.append(f'path="{section.media_path}"')
        if section.media_transcript is not None:
            transcript = section.media_transcript
            # Fence attribute grammar rejects `"` and `\n` at parse
            # time (the `_ATTR_KV_RE` character class is `[^"\n]*`).
            # Refuse to emit unparseable output rather than producing
            # something that survives serialization but fails re-read.
            if '"' in transcript or "\n" in transcript:
                raise ValueError(
                    "AUDIO transcript cannot contain double-quotes or "
                    "newlines — the fence attribute grammar disallows them. "
                    "Use curly quotes ('“'/'”') or rephrase. "
                    f"Offending transcript: {transcript!r}"
                )
            attrs.append(f'transcript="{transcript}"')
        attr_blob = (" " + " ".join(attrs)) if attrs else ""
        fence = f"::{section.type.value}{attr_blob}::\n"
        body = section.content
        if body and not body.endswith("\n"):
            body += "\n"
        return fence + body
    suffix = f"#{section.adapter}" if section.adapter else ""
    fence = f"::{section.type.value}{suffix}::\n"
    body = section.content
    if body and not body.endswith("\n"):
        body += "\n"
    # Schema v7: auto-harvested sections carry a magic-comment marker
    # immediately after the fence. Parser lifts it back into
    # `Section.auto_harvest` + `Section.harvest_source`; emitting it
    # here keeps the round-trip symmetric.
    if section.auto_harvest:
        source = section.harvest_source or ""
        marker = f'<!-- dlm-auto-harvest: source="{source}" -->\n'
        return fence + marker + body
    return fence + body
