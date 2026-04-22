"""v5 → v6 migrator: additive `training.sources` + `training.sources_policy`.

v6 introduces two optional fields that let a `.dlm` declare file-tree
directives instead of inlining every byte. A v5 document without
them parses as v6 unchanged — this migrator is pure identity, present
only to satisfy the migration-framework coverage contract
(`test_all_versions_have_migrator_up_to_latest`).
"""

from __future__ import annotations


def migrate(raw: dict[str, object]) -> dict[str, object]:
    return dict(raw)
