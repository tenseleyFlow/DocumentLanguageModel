"""Resolve `training.sources` directives into synthesized Sections.

Entry point: `expand_sources(parsed, base_path)`. Walks every
`SourceDirective` on the parsed frontmatter, reads matching files
through `dlm.io.text.read_text` (so UTF-8 strict + BOM + CRLF
hygiene are identical to in-body sections), and returns a tuple of
`Section(SectionType.PROSE, ...)` ready to be concatenated with
`parsed.sections` before `build_dataset`.

Content-hash collision defense (sprint risks §1): every synthesized
Section's content is prefixed with a canonical `# source: <relpath>\\n\\n`
header. Two different files with identical bodies therefore produce
distinct `section_id`s — the path becomes part of identity, matching
the file-granular semantics users expect for tree ingestion. The
header costs a handful of tokens per file, which is negligible
against typical file sizes.

Per-source provenance is returned alongside the sections so
`TrainingRunSummary.source_directives` can record what was ingested,
what was skipped, and why.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from dlm.directives.errors import DirectivePathError
from dlm.directives.safety import (
    confine_path,
    enumerate_matching_files,
    is_probably_binary,
)
from dlm.doc.parser import ParsedDlm
from dlm.doc.sections import Section, SectionType
from dlm.io.text import DlmEncodingError, read_text

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceProvenance:
    """Per-directive bookkeeping for the training summary.

    `path` is the directive's raw path (user-facing, before ~ or
    symlink resolution) so it matches the frontmatter on disk.
    `file_count` / `total_bytes` reflect what made it into the
    Section list. `skipped_binary` / `skipped_encoding` /
    `skipped_over_size` break down the drops — if a directive yields
    zero sections, the skip counts let the user see why.
    """

    path: str
    file_count: int
    total_bytes: int
    skipped_binary: int = 0
    skipped_encoding: int = 0
    skipped_over_size: int = 0


@dataclass(frozen=True)
class ExpandResult:
    """Return value of `expand_sources` — sections + provenance."""

    sections: tuple[Section, ...]
    provenance: tuple[SourceProvenance, ...]


def expand_sources(parsed: ParsedDlm, *, base_path: Path) -> ExpandResult:
    """Walk every `training.sources` directive and synthesize sections.

    `base_path` is the `.dlm` file's parent directory — the anchor
    for relative paths and the strict-policy confinement root.
    `parsed.source_path` could supply this but we take it explicitly
    because unit tests commonly synthesize `ParsedDlm` with
    `source_path=None`.

    Returns an `ExpandResult` with an empty sections tuple when the
    frontmatter has no directives — callers can unconditionally
    concatenate without a None check.
    """
    training = parsed.frontmatter.training
    directives = training.sources or ()
    if not directives:
        return ExpandResult(sections=(), provenance=())

    strict = training.sources_policy == "strict"
    sections: list[Section] = []
    provenance: list[SourceProvenance] = []

    for directive in directives:
        root_raw = Path(directive.path)
        # Relative paths anchor on base_path (the .dlm's parent).
        if not root_raw.is_absolute() and not directive.path.startswith("~"):
            root_raw = base_path / root_raw

        resolved_root = confine_path(root_raw, base_path, strict=strict)
        if not resolved_root.exists():
            raise DirectivePathError(resolved_root, "path does not exist")

        dir_sections, dir_prov = _expand_one(
            directive_path=directive.path,
            resolved_root=resolved_root,
            include=directive.include,
            exclude=directive.exclude,
            max_files=directive.max_files,
            max_bytes_per_file=directive.max_bytes_per_file,
        )
        sections.extend(dir_sections)
        provenance.append(dir_prov)

    return ExpandResult(sections=tuple(sections), provenance=tuple(provenance))


def _expand_one(
    *,
    directive_path: str,
    resolved_root: Path,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    max_files: int | None,
    max_bytes_per_file: int | None,
) -> tuple[list[Section], SourceProvenance]:
    """Expand a single directive into sections + per-directive provenance."""
    sections: list[Section] = []
    total_bytes = 0
    skipped_binary = 0
    skipped_encoding = 0
    skipped_over_size = 0

    # Anchor for relpath-in-header. For a single-file directive the
    # header uses the file name; for a directory the relpath is the
    # tree-relative form.
    header_root = resolved_root if resolved_root.is_dir() else resolved_root.parent

    for file_path in enumerate_matching_files(
        resolved_root, include=include, exclude=exclude
    ):
        if max_files is not None and len(sections) >= max_files:
            _LOG.info(
                "directive: hit max_files=%d for %s; truncating deterministically",
                max_files,
                directive_path,
            )
            break

        try:
            size = file_path.stat().st_size
        except OSError as exc:
            _LOG.warning("directive: stat failed for %s: %s; skipping", file_path, exc)
            continue

        if max_bytes_per_file is not None and size > max_bytes_per_file:
            _LOG.info(
                "directive: %s (%d bytes) exceeds max_bytes_per_file=%d; skipping",
                file_path,
                size,
                max_bytes_per_file,
            )
            skipped_over_size += 1
            continue

        try:
            raw = file_path.read_bytes()
        except OSError as exc:
            _LOG.warning("directive: read failed for %s: %s; skipping", file_path, exc)
            continue

        if is_probably_binary(raw):
            _LOG.info(
                "directive: %s looks binary (NUL in first KiB); skipping", file_path
            )
            skipped_binary += 1
            continue

        try:
            text = read_text(file_path)
        except DlmEncodingError:
            _LOG.warning("directive: %s is not UTF-8; skipping", file_path)
            skipped_encoding += 1
            continue

        relpath = file_path.relative_to(header_root).as_posix()
        content = f"# source: {relpath}\n\n{text}"
        sections.append(Section(type=SectionType.PROSE, content=content))
        total_bytes += len(raw)

    return sections, SourceProvenance(
        path=directive_path,
        file_count=len(sections),
        total_bytes=total_bytes,
        skipped_binary=skipped_binary,
        skipped_encoding=skipped_encoding,
        skipped_over_size=skipped_over_size,
    )
