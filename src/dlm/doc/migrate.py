"""In-place frontmatter migration — the write path.

Complements `dlm.doc.versioned.validate_versioned` (the *read* path
used by `parse_file`). The read path migrates in memory and never
touches the source file; the write path is what flips a document's
on-disk `dlm_version` and rewrites the frontmatter.

`migrate_file(path, ...)` is the single entry point. The CLI shell
in `dlm.cli.commands.migrate_cmd` is a thin wrapper over this.

Flow:

1. Read raw text (UTF-8 strict, LF-normalized — the project-wide
   contract from `dlm.io.text.read_text`).
2. Split frontmatter and body on the `---` delimiters.
3. YAML-parse the raw frontmatter into a dict.
4. Run `apply_pending` up to `CURRENT_SCHEMA_VERSION`.
5. If nothing applied → return `[]` (idempotent exit).
6. Otherwise: Pydantic-validate the migrated dict, serialize the new
   frontmatter, join with the original body text verbatim, and atomically
   replace `path` (after writing `<path>.bak` unless `no_backup=True`).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from dlm.doc.errors import FrontmatterError
from dlm.doc.migrations.dispatch import apply_pending
from dlm.doc.schema import CURRENT_SCHEMA_VERSION, DlmFrontmatter
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import serialize
from dlm.io.atomic import write_text
from dlm.io.text import read_text

_FRONTMATTER_DELIM = "---"


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of a `migrate_file` call."""

    path: Path
    applied: list[int]
    target_version: int
    backup_path: Path | None
    wrote: bool


def migrate_file(
    path: Path,
    *,
    dry_run: bool = False,
    no_backup: bool = False,
) -> MigrationResult:
    """Migrate `path` up to `CURRENT_SCHEMA_VERSION`.

    - `dry_run=True` reports what *would* run without writing.
    - `no_backup=True` skips the `<path>.bak` safety copy.

    Returns a `MigrationResult`. `applied=[]` means the document was
    already at or beyond `CURRENT_SCHEMA_VERSION` — a clean no-op.
    """
    text = read_text(path)
    yaml_text, body_text = _split_for_migrate(text, path=path)

    try:
        raw = yaml.safe_load(yaml_text) if yaml_text.strip() else {}
    except yaml.YAMLError as exc:
        raise FrontmatterError(
            f"invalid YAML: {exc}",
            path=path,
            line=2,
        ) from exc

    if not isinstance(raw, dict):
        raise FrontmatterError(
            f"frontmatter must be a mapping, got {type(raw).__name__}",
            path=path,
            line=2,
        )

    migrated, applied = apply_pending(raw, target_version=CURRENT_SCHEMA_VERSION)
    if not applied:
        return MigrationResult(
            path=path,
            applied=[],
            target_version=CURRENT_SCHEMA_VERSION,
            backup_path=None,
            wrote=False,
        )

    # Validate post-migration dict against the current schema so a bad
    # migrator can't silently smear garbage into the document.
    fm = DlmFrontmatter.model_validate(migrated)
    new_text = _rejoin(fm, body_text)

    if dry_run:
        return MigrationResult(
            path=path,
            applied=applied,
            target_version=CURRENT_SCHEMA_VERSION,
            backup_path=None,
            wrote=False,
        )

    backup_path: Path | None = None
    if not no_backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)

    write_text(path, new_text)
    return MigrationResult(
        path=path,
        applied=applied,
        target_version=CURRENT_SCHEMA_VERSION,
        backup_path=backup_path,
        wrote=True,
    )


# --- internals ------------------------------------------------------------


def _split_for_migrate(text: str, *, path: Path) -> tuple[str, str]:
    """Split `text` into (frontmatter_yaml, body_text).

    Mirrors the parser's frontmatter split but does not track body line
    numbers — the body is returned verbatim for rewrite purposes.
    """
    lines = text.split("\n")
    if not lines or lines[0] != _FRONTMATTER_DELIM:
        raise FrontmatterError(
            "expected '---' on line 1 to open frontmatter",
            path=path,
            line=1,
            col=1,
        )
    for i in range(1, len(lines)):
        if lines[i] == _FRONTMATTER_DELIM:
            yaml_text = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :])
            return yaml_text, body
    raise FrontmatterError(
        "no closing '---' found for frontmatter block",
        path=path,
        line=1,
    )


def _rejoin(fm: DlmFrontmatter, body_text: str) -> str:
    """Re-assemble a `.dlm` file from a migrated frontmatter + raw body.

    Preserves the body verbatim (migration never touches section content);
    the serializer is only invoked for the frontmatter header. Ensures a
    single trailing newline on the combined output.
    """
    from dlm.doc.parser import ParsedDlm

    # ParsedDlm serializer emits frontmatter + "\n" + sections. We bypass
    # section serialization by handing an empty sections tuple and
    # concatenating the raw body manually.
    empty = ParsedDlm(frontmatter=fm, sections=_empty_sections())
    header = serialize(empty)  # always ends with "\n"

    # Normalize leading/trailing whitespace on the body to match the
    # canonical layout: exactly one blank line between `---\n` closer
    # and the first body line, and exactly one trailing newline.
    body = body_text.lstrip("\n").rstrip("\n")
    if body:
        return f"{header}\n{body}\n"
    return header


def _empty_sections() -> tuple[Section, ...]:
    """Placeholder tuple for the serializer call; actual body is spliced."""
    _ = SectionType  # imported for typing; unused here
    return ()
