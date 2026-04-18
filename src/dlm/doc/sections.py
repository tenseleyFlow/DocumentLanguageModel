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
from dataclasses import dataclass
from enum import StrEnum

from dlm.io.text import normalize_for_hashing


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
    """

    type: SectionType
    content: str
    start_line: int = 0

    @property
    def section_id(self) -> str:
        """Stable 16-char hex content-hash ID."""
        normalized = normalize_for_hashing(self.content)
        h = hashlib.sha256()
        h.update(self.type.value.encode("utf-8"))
        h.update(b"\n")
        h.update(normalized.encode("utf-8"))
        return h.hexdigest()[: _SECTION_ID_BYTES * 2]
