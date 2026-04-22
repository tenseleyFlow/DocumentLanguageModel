"""v12 → v13 migrator: identity bump for the 2026 registry refresh.

Sprint 40 adds additive base-model registry metadata
(`reasoning_tuned`, `context_length_effective`, `text-moe`) without
changing `.dlm` frontmatter shape. The doc schema still advances to
v13 so migration-aware tooling can distinguish post-refresh docs from
older ones. Existing v12 documents therefore migrate as pure identity.
"""

from __future__ import annotations


def migrate(raw: dict[str, object]) -> dict[str, object]:
    return dict(raw)
