"""Baseline (v1) migrator — intentionally empty.

`CURRENT_SCHEMA_VERSION` is 1 at launch. The first *real* migration
will be `v2.py` when Sprint 18 (or earlier) introduces a new
frontmatter field. This module exists so the dispatcher has a
v1-shaped entry for symmetry; the registry coverage test walks
`range(1, CURRENT_SCHEMA_VERSION)` which is `range(1, 1) == []`, so
no migrator is actually required at v1. We still register an identity
shim so `from_version=1` lookups don't explode the day v2 lands but
v1's migrator was forgotten.

When we do reach `CURRENT_SCHEMA_VERSION=2`, this file migrates v1
documents to v2 (e.g., adds a default value for a new required field
or renames a key).
"""

from __future__ import annotations

# Identity at launch; re-purposed when v2 adds a field. No `@register`
# decoration here because `range(1, 1)` is empty — the coverage test
# would FAIL if a v1 migrator were registered without a matching bump
# to `CURRENT_SCHEMA_VERSION`.


def migrate(raw: dict[str, object]) -> dict[str, object]:
    """Return `raw` unchanged.

    Reserved for the first schema bump — Sprint 18 (DPO → preference
    rename) is the earliest candidate.
    """
    return dict(raw)
