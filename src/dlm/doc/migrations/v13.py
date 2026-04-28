"""v13 → v14 migrator: additive auto-mined preference metadata markers.

Adds additive metadata on ``::preference::`` sections — ``auto_mined``
plus judge provenance / scores / timestamps. These live in body-side
magic-comment markers rather than frontmatter, so existing v13
documents migrate as pure identity. The schema still advances so
migration-aware tooling can distinguish docs written before the mining
loop existed.
"""

from __future__ import annotations


def migrate(raw: dict[str, object]) -> dict[str, object]:
    return dict(raw)
