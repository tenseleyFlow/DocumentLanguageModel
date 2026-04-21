"""v11 → v12 migrator: additive `training.audio` block.

v12 adds the additive `training.audio.auto_resample` knob (default
False). A v11 document parses as v12 unchanged because the new block
defaults match the pre-v12 hard-refusal behavior. Identity migrator —
same shape as v10→v11.
"""

from __future__ import annotations

from dlm.doc.migrations import register


@register(from_version=11)
def migrate(raw: dict[str, object]) -> dict[str, object]:
    return dict(raw)
