"""Pydantic models for the replay corpus.

Two models live here:

- `SectionSnapshot` — the *payload* stored in each zstd frame of
  `corpus.zst`. CBOR-encoded for compact, deterministic binary form.
- `IndexEntry` — the *pointer* stored in `index.json`. JSON for
  human-debuggable storage; sorted by `section_id` on write so
  byte-identical corpora produce byte-identical indexes.

Both are frozen + `extra="forbid"` to match the project's strict-schema
norm. Timestamps are stored tz-naive UTC seconds (microsecond=0) to
match the pattern `dlm.store.manifest` already uses — keeps the JSON
and CBOR representations stable across runs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SECTION_ID_RE: Final = 16  # hex chars


def _utc_now_seconds() -> datetime:
    """Tz-naive UTC with microseconds zeroed — matches manifest.py."""
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


class SectionSnapshot(BaseModel):
    """One section's content + provenance, as stored in `corpus.zst`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str = Field(..., description="16-char content hash (doc.sections).")
    section_type: Literal["prose", "instruction", "preference"]
    content: str = Field(..., description="Raw section body.")
    first_seen_at: datetime = Field(default_factory=_utc_now_seconds)
    last_seen_at: datetime = Field(default_factory=_utc_now_seconds)
    training_runs_seen: list[int] = Field(
        default_factory=list,
        description="Training run IDs that trained on this snapshot.",
    )

    @field_validator("section_id")
    @classmethod
    def _validate_section_id(cls, value: str) -> str:
        if len(value) != _SECTION_ID_RE or not all(c in "0123456789abcdef" for c in value):
            raise ValueError(
                f"section_id must be a 16-char lowercase hex string, got {value!r}"
            )
        return value


class IndexEntry(BaseModel):
    """Per-frame pointer: where to find it + when it was added."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str = Field(..., description="Matches SectionSnapshot.section_id.")
    byte_offset: int = Field(..., ge=0)
    length: int = Field(..., gt=0)
    added_at: datetime = Field(default_factory=_utc_now_seconds)
    weight: float = Field(1.0, ge=0.0, description="Sampler weight floor / override.")

    @field_validator("section_id")
    @classmethod
    def _validate_section_id(cls, value: str) -> str:
        if len(value) != _SECTION_ID_RE or not all(c in "0123456789abcdef" for c in value):
            raise ValueError(
                f"section_id must be a 16-char lowercase hex string, got {value!r}"
            )
        return value
