"""v2 → v3 migrator: additive `training.cpt` block.

v3 introduces the continued-pretraining refinements config:

    training:
      cpt:
        schedule: auto            # auto | dapt | sft
        embed_warmup_steps: 0

All fields default so the migrator is pure identity — v2 documents
parse as v3 with `cpt = CptConfig()` without any rewrite. The
migrator exists only to satisfy the migration-framework coverage
contract: every version in `range(1, CURRENT_SCHEMA_VERSION)` must
have a registered step, even when that step is a no-op.
"""

from __future__ import annotations

from dlm.doc.migrations import register


@register(from_version=2)
def migrate(raw: dict[str, object]) -> dict[str, object]:
    return dict(raw)
