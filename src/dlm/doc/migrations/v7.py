"""v7 → v8 migrator: additive `training.gate` block.

v8 introduces the learned MoE adapter gate (`TrainingConfig.gate`) for
per-prompt routing over `training.adapters`. All fields have defaults
that preserve v7 behavior exactly — `gate.enabled=False` means the
trainer runs identically to v7 and the inference path uses static
weights as before.

A v7 document without `training.gate` parses as v8 unchanged; this is
a pure identity migrator.
"""

from __future__ import annotations


def migrate(raw: dict[str, object]) -> dict[str, object]:
    return dict(raw)
