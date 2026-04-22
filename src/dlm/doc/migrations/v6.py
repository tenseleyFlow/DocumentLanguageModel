"""v6 → v7 migrator: additive `auto_harvest` + `harvest_source` on Section.

v7 introduces two optional body-section fields to record provenance for
sections that were written back into the `.dlm` by `dlm harvest` — the
post-training pull-mode that ingests failing probes from a sway report.
Neither field is part of `section_id`; they're tag-style metadata that
rides alongside the existing `adapter` and `tags` fields.

A v6 document without these fields parses as v7 unchanged — this is a
pure identity migrator.
"""

from __future__ import annotations


def migrate(raw: dict[str, object]) -> dict[str, object]:
    return dict(raw)
