"""v9 → v10 migrator: additive `SectionType.IMAGE` for multi-modal bodies.

v10 introduces the `::image path="..." alt="..."::` fence grammar and
the new `SectionType.IMAGE` enum value. The addition is strictly
body-side — no frontmatter fields move. A v9 document without any
image fences parses as v10 unchanged; this is a pure identity
migrator, same shape as the v7→v8 and v8→v9 additive bumps.
"""

from __future__ import annotations


def migrate(raw: dict[str, object]) -> dict[str, object]:
    return dict(raw)
