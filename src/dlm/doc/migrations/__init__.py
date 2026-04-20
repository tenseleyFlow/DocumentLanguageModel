"""Frontmatter migration registry (Sprint 12b, audit F01/F03).

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
    from dlm.doc.migrations import register

    @register(from_version=N)
    def migrate(raw: dict) -> dict:
        # rewrite `raw` from version N to version N+1
        return {**raw, "new_field": "default"}

Import the new module from this package's `__init__.py` so the
`@register` side-effect runs at import time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

# Map of `from_version` → migrator function. Populated by `@register`
# side-effects when per-version modules are imported below.
MIGRATORS: Final[dict[int, Callable[[dict[str, object]], dict[str, object]]]] = {}


def register(
    from_version: int,
) -> Callable[
    [Callable[[dict[str, object]], dict[str, object]]],
    Callable[[dict[str, object]], dict[str, object]],
]:
    """Decorator: bind a migrator to its `from_version` key.

    Double-registration is a programming bug (two modules claim the
    same migration step) and raises `AssertionError` at import time.
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


# Side-effect imports below run `@register` for each version module.
# Keep imports at the bottom so `register` is defined before they load.
from dlm.doc.migrations import (  # noqa: E402  (imports below `register` defn)
    v1,  # noqa: F401  (side-effect import)
    v2,  # noqa: F401  (side-effect import)
    v3,  # noqa: F401  (side-effect import)
)
