"""v10 → v11 migrator: additive `SectionType.AUDIO` for audio-language bodies.

v11 introduces the `::audio path="..." transcript="..."::` fence
grammar and the new `SectionType.AUDIO` enum value. The addition is
strictly body-side — no frontmatter fields move. A v10 document
without any audio fences parses as v11 unchanged; this is a pure
identity migrator, same shape as the v9→v10 image bump.
"""

from __future__ import annotations

from dlm.doc.migrations import register


@register(from_version=10)
def migrate(raw: dict[str, object]) -> dict[str, object]:
    return dict(raw)
