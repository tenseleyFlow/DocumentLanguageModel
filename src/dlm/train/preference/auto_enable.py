"""Auto-enable DPO when `::preference::` sections exist.

The user-facing rule:

- If the frontmatter sets `training.dpo.enabled` explicitly (either
  `true` or `false`), honor that value.
- Otherwise, if the document contains any `::preference::` sections,
  flip `enabled` to `True` so the orchestrator runs the DPO phase.
- Otherwise, leave `enabled` at its default (`False`).

This mirrors how `use_qlora` is auto-resolved by the hardware plan —
the user can override, but the common case is inferred from content.

We detect "user didn't set this" via pydantic's `model_fields_set`.
When a field is omitted from the input dict, it's not in the set;
when it's present (even if equal to the default), it is. The parser
passes the YAML-derived dict to `TrainingConfig.model_validate`, so
this round-trips correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dlm.doc.sections import Section, SectionType

if TYPE_CHECKING:
    from dlm.doc.schema import DpoConfig


def resolve_dpo_enabled(
    dpo_cfg: DpoConfig,
    sections: list[Section],
) -> DpoConfig:
    """Return `dpo_cfg` with `enabled` resolved against content.

    Returns the original object unchanged when no flip is needed;
    returns a `model_copy(update={"enabled": True})` clone when
    auto-enable fires. Immutability preserved — `DpoConfig` is frozen.
    """
    user_set_enabled = "enabled" in dpo_cfg.model_fields_set
    if user_set_enabled:
        return dpo_cfg

    has_prefs = any(s.type is SectionType.PREFERENCE for s in sections)
    if has_prefs and not dpo_cfg.enabled:
        return dpo_cfg.model_copy(update={"enabled": True})
    return dpo_cfg
