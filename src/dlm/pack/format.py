"""Pydantic models for `.dlm.pack` header + file manifest.

Two models:

- `PackHeader` — top-level `PACK_HEADER.json`. Carries format version,
  creation timestamp, `dlm` tool version, content type (full / no-base /
  no-exports), platform hint, and (when `--include-base` was used on a
  non-redistributable spec) the user's licensee-acceptance URL.
- `PackManifest` — per-file sidecar `manifest.json` that the unpacker
  uses to verify layout + size before running checksum validation.
  `content_sha256` is a deterministic rollup of all file checksums so
  drift is detectable without re-reading every file.

`CURRENT_PACK_FORMAT_VERSION = 1`. Bumps happen only when the on-disk
shape changes in a way a v1-writer reader can't understand; minor
additive fields go in `extras` on the header to avoid a version bump.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

# Load-bearing: the one place the format version lives. `dlm.pack.migrations`
# reads this to enforce its coverage test.
CURRENT_PACK_FORMAT_VERSION: Final[int] = 1

ContentType = Literal["full", "no-base", "no-exports", "minimal"]


class PackHeader(BaseModel):
    """Top-level `PACK_HEADER.json` — the unpacker reads this first.

    `content_type` is a coarse hint about what's inside. `platform_hint`
    is advisory (e.g., `"cuda-sm80+"`, `"mps"`, `"cpu"`); it lets
    `dlm doctor` post-unpack surface "you may need to retrain for this
    host" without blocking the install.

    `licensee_acceptance_url` is populated only when `--include-base`
    was used on a `BaseModelSpec.redistributable=False` model — evidence
    that the packer-user has separate acceptance. `dlm push`
    refuses redistributable=False packs regardless of this field since
    HF redistribution is the problem, not licensing paperwork.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Pack format version 1 is the floor; the migrations framework
    # (`dlm.pack.migrations`) bridges v1 → vN for future bumps. The
    # `ge=1` constraint here means no v0 pack can ever reach the
    # migrator — that's deliberate: v0 predated the
    # format we ship, so there's nothing to migrate from.
    pack_format_version: int = Field(..., ge=1)
    created_at: datetime
    tool_version: str = Field(..., min_length=1)
    content_type: ContentType
    platform_hint: str = Field(..., min_length=1)
    licensee_acceptance_url: str | None = None


class PackManifest(BaseModel):
    """Per-file sidecar — the unpacker reads this to sanity-check the tree.

    `entries` maps relative path → byte size; unpack asserts every
    entry is present before invoking per-file sha256 verification.
    `content_sha256` is `sha256(sorted(f"{relpath}\\n{sha256}\\n"))` —
    a deterministic rollup. Identical inputs always produce identical
    rollups, so two packers writing the same content produce
    byte-identical manifests.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dlm_id: str = Field(..., min_length=1)
    base_model: str = Field(..., min_length=1)
    base_model_revision: str | None = None
    base_model_sha256: str | None = None
    adapter_version: int = Field(..., ge=0)
    entries: dict[str, int] = Field(default_factory=dict)
    content_sha256: str = Field(..., min_length=64, max_length=64)
