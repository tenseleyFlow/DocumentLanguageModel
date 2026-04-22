"""Baseline (v1) pack-format migrator — identity shim.

At launch, `CURRENT_PACK_FORMAT_VERSION = 1`. The first *real* migrator
will be `v2.py` when the pack layout bumps (new subdirectories, schema
changes). This file exists so the registry has a v1-shaped entry for
symmetry.

The coverage test walks `range(1, CURRENT_PACK_FORMAT_VERSION)` — at
v1 that's an empty range, so no migrator is required here. We still
ship an identity function so `from_version=1` lookups don't explode
the day v2 lands but v1's migrator was forgotten.
"""

from __future__ import annotations

from pathlib import Path

# Identity — reserved for future pack layout changes. No `@register`
# decoration until `CURRENT_PACK_FORMAT_VERSION` bumps to 2+, since
# the coverage test would fail if v1 appeared in `PACK_MIGRATORS`
# without a matching bump.


def migrate(root: Path) -> Path:
    """Return `root` unchanged.

    Reserved for the first pack-format bump. Until then the unpacker
    never invokes this; it's part of the public surface so future
    migrators can point downstream at the same shape.
    """
    return root
