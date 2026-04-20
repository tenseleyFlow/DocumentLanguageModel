"""Atomic `dlm.lock` I/O (Sprint 15)."""

from __future__ import annotations

import json
from pathlib import Path

from dlm.io.atomic import write_text
from dlm.lock.errors import LockSchemaError, LockWriteError
from dlm.lock.schema import CURRENT_LOCK_VERSION, LOCK_FILENAME, DlmLock


def lock_path(store_root: Path) -> Path:
    """Return `<store_root>/dlm.lock`. Does not check existence."""
    return store_root / LOCK_FILENAME


def write_lock(store_root: Path, lock: DlmLock) -> Path:
    """Serialize `lock` to `<store_root>/dlm.lock` with byte-stable JSON.

    Uses the repo's atomic-write helper so a crashed write can't leave a
    half-written lock on disk. Returns the path written.
    """
    if lock.lock_version != CURRENT_LOCK_VERSION:
        # Audit-05 N13: programmer error on the write path → dedicated
        # LockWriteError rather than LockSchemaError (which is reserved
        # for read-side parse failures).
        raise LockWriteError(
            path=lock_path(store_root),
            reason=(
                f"lock_version={lock.lock_version!r} != writer's "
                f"CURRENT_LOCK_VERSION={CURRENT_LOCK_VERSION}"
            ),
        )
    target = lock_path(store_root)
    payload = lock.model_dump(mode="json")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    write_text(target, text)
    return target


def load_lock(store_root: Path) -> DlmLock | None:
    """Read `<store_root>/dlm.lock`, return `None` when absent.

    Raises `LockSchemaError` on malformed JSON, unknown version, or
    Pydantic validation failure. A missing file is not an error — it
    means this is a fresh store.
    """
    path = lock_path(store_root)
    if not path.is_file():
        return None

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LockSchemaError(path, f"unreadable: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LockSchemaError(path, f"invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise LockSchemaError(
            path, f"top-level JSON must be an object, got {type(payload).__name__}"
        )

    version = payload.get("lock_version")
    if version != CURRENT_LOCK_VERSION:
        raise LockSchemaError(
            path,
            f"unsupported lock_version {version!r} (reader expects {CURRENT_LOCK_VERSION})",
        )

    try:
        return DlmLock.model_validate(payload)
    except Exception as exc:
        raise LockSchemaError(path, f"schema validation: {exc}") from exc
