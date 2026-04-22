"""Resolve `training.sources` directives into synthesized Sections.

Entry point: `expand_sources(parsed, base_path)`. Walks every
`SourceDirective` on the parsed frontmatter, reads matching files
through `dlm.io.text.read_text` (so UTF-8 strict + BOM + CRLF
hygiene are identical to in-body sections), and returns a tuple of
`Section(SectionType.PROSE, ...)` ready to be concatenated with
`parsed.sections` before `build_dataset`.

Discovery + merge: when a directive points at a directory,
`discover_configs` finds every `.dlm/training.yaml` + `.dlm/ignore`
inside the tree; for each candidate file, `effective_config_for`
resolves the merged include/exclude verdict and the metadata tags to
flow onto the synthesized Section.

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
from typing import Final

from dlm.directives.discovery import DiscoveredConfig, discover_configs
from dlm.directives.errors import DirectivePathError
from dlm.directives.merge import effective_config_for
from dlm.directives.safety import confine_path, is_probably_binary
from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import SourceDirective
from dlm.doc.sections import Section, SectionType
from dlm.io.text import DlmEncodingError, read_text
from dlm.store.blobs import BlobStore

_LOG = logging.getLogger(__name__)

# File extensions dispatched to the blob store as IMAGE sections.
# Kept lowercase; comparison lowers the observed suffix.
_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
)

# File extensions dispatched to the blob store as AUDIO sections.
# `.mp3` / `.m4a` deferred — soundfile can't decode them without
# libsndfile MP3 support, which isn't in our runtime dep tree.
# Users with mp3 corpora re-encode to wav/flac first.
_AUDIO_EXTENSIONS: Final[frozenset[str]] = frozenset({".wav", ".flac", ".ogg"})

# Sidecar transcript filename suffix: `clips/hello.wav` pairs with
# `clips/hello.txt`. Missing the sidecar is a hard refusal — audio
# without text supervision has no training signal, so we surface the
# gap immediately rather than emitting a section that'd fail at
# row-emission time.
_TRANSCRIPT_SIDECAR_SUFFIX: Final[str] = ".txt"


@dataclass(frozen=True)
class SourceProvenance:
    """Per-directive bookkeeping for the training summary.

    `path` is the directive's raw path (user-facing, before ~ or
    symlink resolution) so it matches the frontmatter on disk.
    `file_count` / `total_bytes` reflect text sections that made it
    into the Section list. `image_count` / `image_bytes` /
    `audio_count` / `audio_bytes` reflect media sections ingested
    through the blob store. `skipped_binary` / `skipped_encoding` /
    `skipped_over_size` / `skipped_by_descent` break down the drops —
    if a directive yields zero sections, the skip counts let the user
    see why. `skipped_image_no_store` / `skipped_audio_no_store`
    count media hits dropped because no BlobStore was passed to
    `expand_sources` (tests, `dlm show`). `skipped_audio_no_transcript`
    counts audio files missing their `<stem>.txt` sidecar.
    """

    path: str
    file_count: int
    total_bytes: int
    image_count: int = 0
    image_bytes: int = 0
    audio_count: int = 0
    audio_bytes: int = 0
    skipped_binary: int = 0
    skipped_encoding: int = 0
    skipped_over_size: int = 0
    skipped_by_descent: int = 0
    skipped_image_no_store: int = 0
    skipped_audio_no_store: int = 0
    skipped_audio_no_transcript: int = 0


@dataclass(frozen=True)
class ExpandResult:
    """Return value of `expand_sources` — sections + provenance."""

    sections: tuple[Section, ...]
    provenance: tuple[SourceProvenance, ...]
    discovered: tuple[DiscoveredConfig, ...] = ()


def expand_sources(
    parsed: ParsedDlm,
    *,
    base_path: Path,
    blob_store: BlobStore | None = None,
) -> ExpandResult:
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
    grandparent as the effective base.

    `blob_store` receives ingested image bytes. When `None`, image-
    extension files are counted in `skipped_image_no_store` and no
    IMAGE section is emitted — this is the right shape for read-only
    paths like `dlm show`, where we don't want to mutate disk.

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
            blob_store=blob_store,
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
    blob_store: BlobStore | None,
) -> tuple[list[Section], SourceProvenance]:
    """Expand a single directive into sections + per-directive provenance."""
    sections: list[Section] = []
    total_bytes = 0
    image_count = 0
    image_bytes = 0
    audio_count = 0
    audio_bytes = 0
    skipped_binary = 0
    skipped_encoding = 0
    skipped_over_size = 0
    skipped_by_descent = 0
    skipped_image_no_store = 0
    skipped_audio_no_store = 0
    skipped_audio_no_transcript = 0

    # Anchor for relpath-in-header. For a single-file directive the
    # header uses the file name; for a directory the relpath is the
    # tree-relative form.
    header_root = resolved_root if resolved_root.is_dir() else resolved_root.parent

    for file_path in _iter_candidates(resolved_root):
        if directive.max_files is not None and _section_cap_reached(sections, directive.max_files):
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

        # Image-extension dispatch: skips the text-read + binary-skip
        # path entirely. The blob store owns ingestion; the synthesized
        # section carries only the path + blob sha.
        if file_path.suffix.lower() in _IMAGE_EXTENSIONS:
            if blob_store is None:
                _LOG.info(
                    "directive: %s is an image but no blob_store supplied; skipping",
                    file_path,
                )
                skipped_image_no_store += 1
                continue
            handle = blob_store.put(file_path)
            relpath = file_path.relative_to(header_root).as_posix()
            alt = file_path.stem
            sections.append(
                Section(
                    type=SectionType.IMAGE,
                    content="",
                    media_path=relpath,
                    media_alt=alt,
                    media_blob_sha=handle.sha,
                    tags=effective.tags,
                ),
            )
            image_count += 1
            image_bytes += handle.size
            continue

        # Audio-extension dispatch: same shape as image but requires
        # a `<stem>.txt` sidecar for the transcript —
        # audio without text has no training signal. Missing sidecar
        # is a skip (with an explicit counter), not a hard raise,
        # because a mixed corpus may have both "for-training" audio
        # (has .txt) and "reference" audio (no .txt) side by side.
        if file_path.suffix.lower() in _AUDIO_EXTENSIONS:
            if blob_store is None:
                _LOG.info(
                    "directive: %s is audio but no blob_store supplied; skipping",
                    file_path,
                )
                skipped_audio_no_store += 1
                continue
            transcript = _read_audio_transcript(file_path)
            if transcript is None:
                _LOG.info(
                    "directive: %s has no %s sidecar; skipping "
                    "(audio without transcript has no training signal)",
                    file_path,
                    _TRANSCRIPT_SIDECAR_SUFFIX,
                )
                skipped_audio_no_transcript += 1
                continue
            handle = blob_store.put(file_path)
            relpath = file_path.relative_to(header_root).as_posix()
            sections.append(
                Section(
                    type=SectionType.AUDIO,
                    content="",
                    media_path=relpath,
                    media_blob_sha=handle.sha,
                    media_transcript=transcript,
                    tags=effective.tags,
                ),
            )
            audio_count += 1
            audio_bytes += handle.size
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

    # Count prose-bearing sections separately from media so the
    # per-modality counters don't collide. PROSE + INSTRUCTION +
    # PREFERENCE → file_count; IMAGE → image_count; AUDIO →
    # audio_count.
    text_sections = sum(1 for s in sections if s.type not in (SectionType.IMAGE, SectionType.AUDIO))
    return sections, SourceProvenance(
        path=directive.path,
        file_count=text_sections,
        total_bytes=total_bytes,
        image_count=image_count,
        image_bytes=image_bytes,
        audio_count=audio_count,
        audio_bytes=audio_bytes,
        skipped_binary=skipped_binary,
        skipped_encoding=skipped_encoding,
        skipped_over_size=skipped_over_size,
        skipped_by_descent=skipped_by_descent,
        skipped_image_no_store=skipped_image_no_store,
        skipped_audio_no_store=skipped_audio_no_store,
        skipped_audio_no_transcript=skipped_audio_no_transcript,
    )


def _read_audio_transcript(audio_path: Path) -> str | None:
    """Read the sibling `<stem>.txt` transcript, or return None.

    Sidecar is `<stem><_TRANSCRIPT_SIDECAR_SUFFIX>` in the same
    directory as the audio file — `clips/hello.wav` pairs with
    `clips/hello.txt`. The transcript is stripped of leading/trailing
    whitespace and re-UTF-8-decoded through `dlm.io.text.read_text`
    so it matches the encoding contract the rest of the pipeline
    uses for ingested text.

    Returns None (not empty string) when the sidecar is missing — the
    caller treats "no sidecar" and "empty sidecar" differently: the
    former is a skip, the latter is an author bug surfaced at train
    time via a loud refusal (an empty transcript has no training
    signal either).
    """
    sidecar = audio_path.with_suffix(_TRANSCRIPT_SIDECAR_SUFFIX)
    if not sidecar.exists():
        return None
    try:
        text = read_text(sidecar)
    except (OSError, DlmEncodingError) as exc:
        _LOG.warning(
            "directive: transcript sidecar %s is unreadable: %s; skipping audio",
            sidecar,
            exc,
        )
        return None
    return text.strip()


def _section_cap_reached(sections: list[Section], max_files: int) -> bool:
    """True when the cap has been hit for the user's intent.

    `max_files` caps total ingested files (text + image) — the user
    wrote one number; both modalities count against it.
    """
    return len(sections) >= max_files


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
