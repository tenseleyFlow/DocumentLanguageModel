"""Resolve `training.sources` directives into synthesized Sections.

Entry point: `expand_sources(parsed, base_path)`. Walks every
`SourceDirective` on the parsed frontmatter, reads matching files
through `dlm.io.text.read_text` (so UTF-8 strict + BOM + CRLF
hygiene are identical to in-body sections), and returns a tuple of
`Section(SectionType.PROSE, ...)` ready to be concatenated with
`parsed.sections` before `build_dataset`.

Discovery + merge (Sprint 30): when a directive points at a
directory, `discover_configs` finds every `.dlm/training.yaml` +
`.dlm/ignore` inside the tree; for each candidate file,
`effective_config_for` resolves the merged include/exclude verdict
and the metadata tags to flow onto the synthesized Section.

Content-hash collision defense: every synthesized Section's content
is prefixed with a canonical `# source: <relpath>\\n\\n` header. Two
different files with identical bodies therefore produce distinct
`section_id`s — the path becomes part of identity, matching the
file-granular semantics users expect for tree ingestion. Tags are
not part of `section_id`, so metadata churn doesn't invalidate the
replay corpus.

Per-source provenance is returned alongside the sections so
`TrainingRunSummary.source_directives` can record what was ingested,
what was skipped, and why.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from dlm.directives.discovery import DiscoveredConfig, discover_configs
from dlm.directives.errors import DirectivePathError
from dlm.directives.merge import effective_config_for
from dlm.directives.safety import confine_path, is_probably_binary
from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import SourceDirective
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
    `skipped_over_size` / `skipped_by_descent` break down the drops —
    if a directive yields zero sections, the skip counts let the user
    see why.
    """

    path: str
    file_count: int
    total_bytes: int
    skipped_binary: int = 0
    skipped_encoding: int = 0
    skipped_over_size: int = 0
    skipped_by_descent: int = 0


@dataclass(frozen=True)
class ExpandResult:
    """Return value of `expand_sources` — sections + provenance."""

    sections: tuple[Section, ...]
    provenance: tuple[SourceProvenance, ...]
    discovered: tuple[DiscoveredConfig, ...] = ()


def expand_sources(parsed: ParsedDlm, *, base_path: Path) -> ExpandResult:
    """Walk every `training.sources` directive and synthesize sections.

    `base_path` is the `.dlm` file's parent directory — the anchor
    for relative paths and the strict-policy confinement root.
    `parsed.source_path` could supply this but we take it explicitly
    because unit tests commonly synthesize `ParsedDlm` with
    `source_path=None`.

    When the `.dlm` lives under a `.dlm/` metadata directory (the
    scaffolded shape at `<corpus>/.dlm/corpus.dlm`), the user's
    intended anchor is the corpus directory, not the metadata
    directory — so relative resolution and strict confinement use the
    grandparent as the effective base. Fixes Audit-09 B2.

    Returns an `ExpandResult` with an empty sections tuple when the
    frontmatter has no directives — callers can unconditionally
    concatenate without a None check.
    """
    training = parsed.frontmatter.training
    directives = training.sources or ()
    if not directives:
        return ExpandResult(sections=(), provenance=(), discovered=())

    effective_base = base_path.parent if base_path.name == ".dlm" else base_path
    strict = training.sources_policy == "strict"
    sections: list[Section] = []
    provenance: list[SourceProvenance] = []
    all_discovered: list[DiscoveredConfig] = []

    for directive in directives:
        root_raw = Path(directive.path)
        # Relative paths anchor on effective_base (the corpus dir for
        # scaffolded .dlms, the .dlm's parent otherwise).
        if not root_raw.is_absolute() and not directive.path.startswith("~"):
            root_raw = effective_base / root_raw

        resolved_root = confine_path(root_raw, effective_base, strict=strict)
        if not resolved_root.exists():
            raise DirectivePathError(resolved_root, "path does not exist")

        # Discovery: only meaningful for directory directives. A
        # single-file directive has no tree to descend into.
        discovered = discover_configs(resolved_root) if resolved_root.is_dir() else ()
        all_discovered.extend(discovered)

        dir_sections, dir_prov = _expand_one(
            directive=directive,
            resolved_root=resolved_root,
            discovered=discovered,
        )
        sections.extend(dir_sections)
        provenance.append(dir_prov)

    return ExpandResult(
        sections=tuple(sections),
        provenance=tuple(provenance),
        discovered=tuple(all_discovered),
    )


def _expand_one(
    *,
    directive: SourceDirective,
    resolved_root: Path,
    discovered: tuple[DiscoveredConfig, ...],
) -> tuple[list[Section], SourceProvenance]:
    """Expand a single directive into sections + per-directive provenance."""
    sections: list[Section] = []
    total_bytes = 0
    skipped_binary = 0
    skipped_encoding = 0
    skipped_over_size = 0
    skipped_by_descent = 0

    # Anchor for relpath-in-header. For a single-file directive the
    # header uses the file name; for a directory the relpath is the
    # tree-relative form.
    header_root = resolved_root if resolved_root.is_dir() else resolved_root.parent

    for file_path in _iter_candidates(resolved_root):
        if directive.max_files is not None and len(sections) >= directive.max_files:
            _LOG.info(
                "directive: hit max_files=%d for %s; truncating deterministically",
                directive.max_files,
                directive.path,
            )
            break

        # Descent verdict: merge parent directive + discovered `.dlm/`
        # configs. `included=False` means a deeper config or default-
        # exclude ruled this file out.
        effective = effective_config_for(
            file_path,
            source_root=resolved_root,
            discovered=discovered,
            parent_directive=directive,
            is_dir=False,
        )
        if not effective.included:
            skipped_by_descent += 1
            continue

        try:
            size = file_path.stat().st_size
        except OSError as exc:
            _LOG.warning("directive: stat failed for %s: %s; skipping", file_path, exc)
            continue

        if directive.max_bytes_per_file is not None and size > directive.max_bytes_per_file:
            _LOG.info(
                "directive: %s (%d bytes) exceeds max_bytes_per_file=%d; skipping",
                file_path,
                size,
                directive.max_bytes_per_file,
            )
            skipped_over_size += 1
            continue

        try:
            raw = file_path.read_bytes()
        except OSError as exc:
            _LOG.warning("directive: read failed for %s: %s; skipping", file_path, exc)
            continue

        if is_probably_binary(raw):
            _LOG.info("directive: %s looks binary (NUL in first KiB); skipping", file_path)
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
        sections.append(Section(type=SectionType.PROSE, content=content, tags=effective.tags))
        total_bytes += len(raw)

    return sections, SourceProvenance(
        path=directive.path,
        file_count=len(sections),
        total_bytes=total_bytes,
        skipped_binary=skipped_binary,
        skipped_encoding=skipped_encoding,
        skipped_over_size=skipped_over_size,
        skipped_by_descent=skipped_by_descent,
    )


def _iter_candidates(root: Path) -> Iterator[Path]:
    """Yield every file under `root` in deterministic order.

    Filtering moves into `effective_config_for` now that descent
    protocol layers rules beyond the parent directive's include /
    exclude. Single-file roots are yielded as the single file.
    """
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file():
            yield candidate
