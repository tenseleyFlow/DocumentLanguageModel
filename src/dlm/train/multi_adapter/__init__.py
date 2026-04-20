"""Multi-adapter composition: name-keyed LoRA adapters in one document.

Two surfaces: `router.rows_for(parsed, name)` filters the document's
sections down to the rows that train adapter `name`, and (in 20b) a
trainer orchestrator drives one-adapter-at-a-time training over the
named set. The module is isolated so the single-adapter path stays
untouched when `training.adapters` is absent.
"""
