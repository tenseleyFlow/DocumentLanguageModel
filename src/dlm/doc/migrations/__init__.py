"""Frontmatter migration registry.

Each `dlm_version` bump registers one module under this package,
exporting a `migrate(raw: dict) -> dict` function that rewrites the
raw YAML dict from version N to version N+1. The dispatcher
(`dispatch.apply_pending`) chains them to walk an old document up to
`CURRENT_SCHEMA_VERSION`.

Enforcement contract: `test_all_versions_have_migrator_up_to_latest`
walks this registry and refuses any CI run where
`set(MIGRATORS) != set(range(1, CURRENT_SCHEMA_VERSION))`. A PR that
bumps the schema version without registering a matching migrator fails
that test, so the rule is statically enforced.

Minimum viable registration:

    # src/dlm/doc/migrations/vN.py
    def migrate(raw: dict) -> dict:
        # rewrite `raw` from version N to version N+1
        return {**raw, "new_field": "default"}

    # src/dlm/doc/migrations/__init__.py
    MIGRATORS[N] = vN.migrate
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

# Keep the per-version modules imported here so the shipped registry is
# declared in one explicit place rather than assembled via import-time
# side effects.
from dlm.doc.migrations import (
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    v8,
    v9,
    v10,
    v11,
    v12,
    v13,
)

# Map of `from_version` → migrator function for the shipped schema path.
MIGRATORS: Final[dict[int, Callable[[dict[str, object]], dict[str, object]]]] = {
    1: v1.migrate,
    2: v2.migrate,
    3: v3.migrate,
    4: v4.migrate,
    5: v5.migrate,
    6: v6.migrate,
    7: v7.migrate,
    8: v8.migrate,
    9: v9.migrate,
    10: v10.migrate,
    11: v11.migrate,
    12: v12.migrate,
    13: v13.migrate,
}


def register(
    from_version: int,
) -> Callable[
    [Callable[[dict[str, object]], dict[str, object]]],
    Callable[[dict[str, object]], dict[str, object]],
]:
    """Test helper: bind a temporary migrator to a `from_version` key.

    Runtime migrators are declared explicitly in `MIGRATORS` above.
    Tests still use this helper to swap in synthetic upgrade chains
    without editing the shipped modules.
    """

    def decorator(
        fn: Callable[[dict[str, object]], dict[str, object]],
    ) -> Callable[[dict[str, object]], dict[str, object]]:
        assert from_version not in MIGRATORS, (
            f"duplicate migrator for from_version={from_version}: "
            f"{MIGRATORS[from_version].__module__} vs {fn.__module__}"
        )
        MIGRATORS[from_version] = fn
        return fn

    return decorator
