"""Migration dispatcher — walk a raw frontmatter dict up to target version.

Call path:

    raw = yaml.safe_load(frontmatter_text)
    migrated, applied = apply_pending(raw, target_version=CURRENT_SCHEMA_VERSION)
    # `migrated` is now shape-compatible with the Pydantic model;
    # `applied` is the list of from-versions that ran, e.g., [1, 2].

The dispatcher stamps `raw["dlm_version"] = v+1` after each successful
migrator so any migrator whose output is inspected mid-chain sees the
running version. Already-current (or newer) inputs return the original
raw with `applied=[]`.
"""

from __future__ import annotations

from dlm.doc.errors import UnsupportedMigrationError
from dlm.doc.migrations import MIGRATORS


def apply_pending(
    raw: dict[str, object], *, target_version: int
) -> tuple[dict[str, object], list[int]]:
    """Walk `raw` through the migrator chain up to `target_version`.

    Returns `(migrated_raw, applied_from_versions)`. `applied` is the
    list of `from_version` integers that ran, in order — empty when the
    input was already at or above `target_version`.

    Raises `UnsupportedMigrationError` when a `from_version` in the
    chain has no migrator registered (the "manually-forked" case; the
    coverage test catches missing migrators at commit time).
    """
    current = dict(raw)
    applied: list[int] = []

    while True:
        version = current.get("dlm_version", 1)
        if not isinstance(version, int):
            raise UnsupportedMigrationError(
                f"dlm_version must be int, got {type(version).__name__}",
            )
        if version >= target_version:
            return current, applied
        migrator = MIGRATORS.get(version)
        if migrator is None:
            raise UnsupportedMigrationError(
                f"no migrator registered for dlm_version={version}; "
                f"target_version={target_version}.",
            )
        current = migrator(current)
        current["dlm_version"] = version + 1
        applied.append(version)
