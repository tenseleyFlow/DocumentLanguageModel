"""v4 → v5 migrator: additive `training.precision` override.

v5 introduces an optional `training.precision` field that lets users
opt into fp16/bf16 on MPS (where the auto-picker now defaults to fp32
after the NaN-adapter bug of 2026-04-20; see
`.docs/bugs/01-nan-adapter-on-mps.md`). A v4 document without the
field parses as v5 unchanged — this migrator is pure identity,
present only to satisfy the migration-framework coverage contract.
"""

from __future__ import annotations

from dlm.doc.migrations import register


@register(from_version=4)
def migrate(raw: dict[str, object]) -> dict[str, object]:
    return dict(raw)
