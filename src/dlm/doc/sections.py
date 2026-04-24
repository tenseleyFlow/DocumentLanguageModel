"""Body sections: PROSE (default), INSTRUCTION (`::instruction::`),
PREFERENCE (`::preference::`), IMAGE (`::image path="..." alt="..."::`),
AUDIO (`::audio path="..." transcript="..."::`).

Text-body sections carry their raw content verbatim plus a stable
`section_id` derived from `sha256(type || "\\n" || normalized_content)[:16]`
where normalization is the same LF+BOM-stripping applied by `dlm.io.text`.

Media sections (IMAGE, AUDIO) reference a binary blob outside the
`.dlm` file; their identity is
`sha256(type || "\\n" || path || "\\n" || blob_sha)[:16]` once the blob
has been ingested into the content-addressed store. The path is part
of the hash because different logical uses of the same bytes
(`hero.png` in section A, `same-bytes.png` in section B) should not
collapse to one training row. Before ingestion, `media_blob_sha` is
`None` and the path alone seeds identity — sufficient for `dlm show`
but not for training.

This means:

- The section ID is stable across Windows/Unix line endings (audit F15).
- A whitespace-only edit inside *another* section does not change this
  section's ID (content-addressing correctness).
- Changing the section type (prose → instruction, image → audio)
  produces a different ID even for identical content (type namespaces
  are disjoint).
- For media sections, a blob-bytes change flips the ID even if the
  path didn't move; a path change flips the ID even if the bytes are
  identical.

AUDIO vs IMAGE: audio sections require `media_transcript` (text-side
supervision); image sections optionally carry a caption in `content`.
The training row for audio ties the transcript to the audio features;
the training row for image uses the image-token placeholder and caption.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from dlm.io.text import normalize_for_hashing

_EMPTY_TAGS: Mapping[str, str] = MappingProxyType({})


class SectionType(StrEnum):
    PROSE = "prose"
    INSTRUCTION = "instruction"
    PREFERENCE = "preference"
    IMAGE = "image"
    AUDIO = "audio"


_SECTION_ID_BYTES = 8  # 16 hex chars


@dataclass(frozen=True)
class Section:
    """A single body section.

    `start_line` is the 1-indexed line in the source where the section
    begins (the fence line for fenced sections, the first prose line for
    PROSE). Used for error reporting and is **not** part of the section
    identity.

    `content` is the raw section body, fence-free. Fence lines are
    stripped; leading/trailing blank lines around the content are
    preserved as-is to keep round-trip idempotent after the first pass.

    `adapter` is the optional `#name` routing suffix from a fence like
    `::instruction#tone::`. `None` means "unrouted" — the section's rows
    flow to whichever adapter the router picks as default (the first
    declared, in multi-adapter docs). The field is intentionally not
    part of `section_id`: moving a section between adapters is a routing
    change, not a content change, and retention snapshots key off the
    content hash.

    `tags` is the optional free-form metadata map flowed from
    `.dlm/training.yaml`. Consumers (weighting, filtering,
    sway probes) read these; the trainer's row-production path
    ignores them. Like `adapter`, tags are **not** part of `section_id`
    — metadata churn doesn't invalidate replay identity.

    `auto_harvest` marks a section as written back into the `.dlm` by
    `dlm harvest` — the pull-mode that ingests failing probes from a
    sway report (schema v7). `harvest_source` records the source run
    ("run_N_sway"-style opaque token) for provenance. Like `tags`,
    neither field participates in `section_id`.

    `auto_mined` marks a `::preference::` section as synthesized by
    Sprint 42's preference-mining loop rather than hand-authored. The
    accompanying judge metadata (`judge_name`, `judge_score_chosen`,
    `judge_score_rejected`, `mined_at`, `mined_run_id`) captures
    provenance for review, metrics, and revert flows. Like harvest
    metadata, these fields do not participate in `section_id`.

    `auto_synth` marks an `::instruction::` section as synthesized by
    Sprint 43's instruction-generation loop rather than hand-authored.
    The accompanying metadata (`synth_teacher`, `synth_strategy`,
    `synth_at`, `source_section_id`) captures provenance for review,
    metrics, and revert flows. Like the other provenance flags, these
    fields do not participate in `section_id`.

    `media_path` / `media_alt` / `media_blob_sha` are media-section
    fields (IMAGE + AUDIO) populated from the fence attributes and
    the content-addressed blob store (after ingestion). Non-media
    sections leave them at their `None` defaults and they do not
    participate in identity; media sections use them as the identity
    inputs in place of `content`. `media_alt` is IMAGE-only;
    `media_transcript` is AUDIO-only (the audio's text-side
    supervision, required for training).
    """

    type: SectionType
    content: str
    start_line: int = 0
    adapter: str | None = None
    tags: Mapping[str, str] = field(default_factory=lambda: _EMPTY_TAGS)
    auto_harvest: bool = False
    harvest_source: str | None = None
    auto_mined: bool = False
    judge_name: str | None = None
    judge_score_chosen: float | None = None
    judge_score_rejected: float | None = None
    mined_at: str | None = None
    mined_run_id: int | None = None
    auto_synth: bool = False
    synth_teacher: str | None = None
    synth_strategy: str | None = None
    synth_at: str | None = None
    source_section_id: str | None = None
    media_path: str | None = None
    media_alt: str | None = None
    media_blob_sha: str | None = None
    media_transcript: str | None = None

    def __post_init__(self) -> None:
        if self.auto_mined:
            if self.type != SectionType.PREFERENCE:
                raise ValueError("auto_mined metadata is only valid on preference sections")
            missing = [
                name
                for name, value in (
                    ("judge_name", self.judge_name),
                    ("judge_score_chosen", self.judge_score_chosen),
                    ("judge_score_rejected", self.judge_score_rejected),
                    ("mined_at", self.mined_at),
                    ("mined_run_id", self.mined_run_id),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"auto_mined preference sections require metadata fields {missing!r}"
                )
            assert self.judge_score_chosen is not None
            assert self.judge_score_rejected is not None
            if not math.isfinite(self.judge_score_chosen) or not math.isfinite(
                self.judge_score_rejected
            ):
                raise ValueError("judge scores must be finite floats")
            assert self.mined_run_id is not None
            if self.mined_run_id < 1:
                raise ValueError("mined_run_id must be >= 1")

        if self.auto_synth:
            if self.type != SectionType.INSTRUCTION:
                raise ValueError("auto_synth metadata is only valid on instruction sections")
            missing = [
                name
                for name, value in (
                    ("synth_teacher", self.synth_teacher),
                    ("synth_strategy", self.synth_strategy),
                    ("synth_at", self.synth_at),
                    ("source_section_id", self.source_section_id),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"auto_synth instruction sections require metadata fields {missing!r}"
                )
            assert self.synth_teacher is not None
            assert self.synth_strategy is not None
            assert self.synth_at is not None
            assert self.source_section_id is not None
            if not self.synth_teacher:
                raise ValueError("synth_teacher must be non-empty")
            if not self.synth_strategy:
                raise ValueError("synth_strategy must be non-empty")
            if len(self.source_section_id) != _SECTION_ID_BYTES * 2 or any(
                ch not in "0123456789abcdef" for ch in self.source_section_id
            ):
                raise ValueError("source_section_id must be a 16-char lowercase hex section id")

    @property
    def section_id(self) -> str:
        """Stable 16-char hex content-hash ID."""
        h = hashlib.sha256()
        h.update(self.type.value.encode("utf-8"))
        h.update(b"\n")
        if self.type in (SectionType.IMAGE, SectionType.AUDIO):
            # Media identity: path || blob_sha. Pre-ingest fallback
            # hashes path alone so `dlm show` and parser round-trips
            # work before the trainer writes bytes through the blob
            # store; the trainer always populates `media_blob_sha`
            # before deterministic splits see the ID. Transcript /
            # alt-text do not participate — they're metadata on the
            # section, not part of identity (edit an audio's
            # transcript → same section, new training pair; edit the
            # audio bytes → new section entirely).
            h.update((self.media_path or "").encode("utf-8"))
            if self.media_blob_sha is not None:
                h.update(b"\n")
                h.update(self.media_blob_sha.encode("utf-8"))
        else:
            normalized = normalize_for_hashing(self.content)
            h.update(normalized.encode("utf-8"))
        return h.hexdigest()[: _SECTION_ID_BYTES * 2]
