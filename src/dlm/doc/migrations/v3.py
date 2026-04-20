"""v3 → v4 migrator: additive `training.adapters` block.

v4 introduces named multi-adapter composition:

    training:
      adapters:
        knowledge: {adapter: lora, lora_r: 8, ...}
        tone:      {adapter: lora, lora_r: 4, ...}

The flat `adapter`/`lora_*` keys stay the single-adapter shorthand and
are not rewritten. A v3 document without `adapters` parses as v4
unchanged — this migrator is pure identity, present only to satisfy
the migration-framework coverage contract.
"""

from __future__ import annotations

from dlm.doc.migrations import register


@register(from_version=3)
def migrate(raw: dict[str, object]) -> dict[str, object]:
    return dict(raw)
