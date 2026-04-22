"""v8 → v9 migrator: additive `training.cache` block.

v9 introduces per-document tokenized-cache tuning
(`TrainingConfig.cache`) — three knobs (`enabled`, `max_bytes`,
`prune_older_than_days`) that let a `.dlm` override the pre-v9
hard-coded defaults. All fields carry defaults that preserve v8
behavior exactly.

A v8 document without `training.cache` parses as v9 unchanged; this
is a pure identity migrator, same shape as the v7→v8 gate migrator.
"""

from __future__ import annotations


def migrate(raw: dict[str, object]) -> dict[str, object]:
    return dict(raw)
