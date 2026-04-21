"""Body sections: PROSE (default), INSTRUCTION (`::instruction::`),
PREFERENCE (`::preference::`).

Each section carries its raw content verbatim plus a stable `section_id`
derived from `sha256(type || "\\n" || normalized_content)[:16]` where
normalization is the same LF+BOM-stripping applied by `dlm.io.text`.

This means:

- The section ID is stable across Windows/Unix line endings (audit F15).
- A whitespace-only edit inside *another* section does not change this
  section's ID (content-addressing correctness).
- Changing the section type (prose → instruction) produces a different ID
  even for identical content.
"""

from __future__ import annotations

import hashlib
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
    `.dlm/training.yaml` (Sprint 30). Consumers (weighting, filtering,
    sway probes) read these; the trainer's row-production path
    ignores them. Like `adapter`, tags are **not** part of `section_id`
    — metadata churn doesn't invalidate replay identity.

    `auto_harvest` marks a section as written back into the `.dlm` by
    `dlm harvest` — the pull-mode that ingests failing probes from a
    sway report (schema v7). `harvest_source` records the source run
    ("run_N_sway"-style opaque token) for provenance. Like `tags`,
    neither field participates in `section_id`.
    """

    type: SectionType
    content: str
    start_line: int = 0
    adapter: str | None = None
    tags: Mapping[str, str] = field(default_factory=lambda: _EMPTY_TAGS)
    auto_harvest: bool = False
    harvest_source: str | None = None

    @property
    def section_id(self) -> str:
        """Stable 16-char hex content-hash ID."""
        normalized = normalize_for_hashing(self.content)
        h = hashlib.sha256()
        h.update(self.type.value.encode("utf-8"))
        h.update(b"\n")
        h.update(normalized.encode("utf-8"))
        return h.hexdigest()[: _SECTION_ID_BYTES * 2]
