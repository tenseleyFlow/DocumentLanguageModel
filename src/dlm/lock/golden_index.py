"""Repo-level index of checked-in determinism goldens.

Separate from the per-store `dlm.lock`: this file tracks which
runtime tuples have an approved golden under `tests/golden/determinism/`.
The canonical path is `.determinism/lock.json`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from dlm.io.atomic import write_text
from dlm.lock.errors import GoldenIndexSchemaError, GoldenIndexWriteError

GOLDEN_INDEX_RELATIVE_PATH: Final[str] = ".determinism/lock.json"
CURRENT_GOLDEN_INDEX_VERSION: Final[int] = 1


class DeterminismGoldenEntry(BaseModel):
    """One approved tuple golden tracked at repo scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    golden_relpath: str = Field(
        ...,
        pattern=r"^tests/golden/determinism/tuple-[0-9a-f]{16}\.json$",
    )
    adapter_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    platform: str = Field(..., min_length=1)
    pinned_versions: dict[str, str] = Field(default_factory=dict)


class DeterminismGoldenIndex(BaseModel):
    """Checked-in set of approved determinism goldens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lock_version: int = Field(CURRENT_GOLDEN_INDEX_VERSION, ge=1)
    updated_at: datetime
    goldens: tuple[DeterminismGoldenEntry, ...] = ()


def golden_index_path(repo_root: Path) -> Path:
    """Return `<repo_root>/.determinism/lock.json`."""

    return repo_root / GOLDEN_INDEX_RELATIVE_PATH


def write_golden_index(repo_root: Path, index: DeterminismGoldenIndex) -> Path:
    """Atomically persist the repo-level determinism-golden index."""

    target = golden_index_path(repo_root)
    if index.lock_version != CURRENT_GOLDEN_INDEX_VERSION:
        raise GoldenIndexWriteError(
            path=target,
            reason=(
                f"lock_version={index.lock_version!r} != writer's "
                f"CURRENT_GOLDEN_INDEX_VERSION={CURRENT_GOLDEN_INDEX_VERSION}"
            ),
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = index.model_dump(mode="json")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    write_text(target, text)
    return target


def load_golden_index(repo_root: Path) -> DeterminismGoldenIndex | None:
    """Read `.determinism/lock.json`, returning `None` when absent."""

    path = golden_index_path(repo_root)
    if not path.is_file():
        return None

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GoldenIndexSchemaError(path, f"unreadable: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GoldenIndexSchemaError(path, f"invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise GoldenIndexSchemaError(
            path,
            f"top-level JSON must be an object, got {type(payload).__name__}",
        )

    version = payload.get("lock_version")
    if version != CURRENT_GOLDEN_INDEX_VERSION:
        raise GoldenIndexSchemaError(
            path,
            f"unsupported lock_version {version!r} (reader expects {CURRENT_GOLDEN_INDEX_VERSION})",
        )

    try:
        return DeterminismGoldenIndex.model_validate(payload)
    except Exception as exc:
        raise GoldenIndexSchemaError(path, f"schema validation: {exc}") from exc


def upsert_golden_index(
    repo_root: Path,
    *,
    golden_relpath: str,
    adapter_sha256: str,
    platform: str,
    pinned_versions: Mapping[str, str],
) -> Path:
    """Insert or replace one tuple golden in `.determinism/lock.json`."""

    current = load_golden_index(repo_root)
    entries = {} if current is None else {entry.golden_relpath: entry for entry in current.goldens}
    entries[golden_relpath] = DeterminismGoldenEntry(
        golden_relpath=golden_relpath,
        adapter_sha256=adapter_sha256,
        platform=platform,
        pinned_versions=dict(sorted(pinned_versions.items())),
    )
    updated = DeterminismGoldenIndex(
        updated_at=_utcnow(),
        goldens=tuple(sorted(entries.values(), key=lambda entry: entry.golden_relpath)),
    )
    return write_golden_index(repo_root, updated)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)
