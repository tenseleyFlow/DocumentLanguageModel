"""v14 → v15 migrator: identity bump for auto-synth instruction metadata.

Adds additive metadata on ``::instruction::`` sections — ``auto_synth``
plus teacher / strategy / timestamp / source provenance. These live in
body-side magic-comment markers rather than frontmatter, so existing
v14 documents migrate as pure identity. The schema still advances so
migration-aware tooling can distinguish docs written before the synth
loop existed.
"""

from __future__ import annotations


def migrate(raw: dict[str, object]) -> dict[str, object]:
    return dict(raw)
