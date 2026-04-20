"""v1 → v2 migrator: `training.dpo` → `training.preference`.

v1 (flat):

    training:
      dpo:
        enabled: true
        beta: 0.1
        loss_type: sigmoid
        learning_rate: 5e-6
        num_epochs: 1
        reference: pre_dpo_adapter

v2 (method-switched + grouped hyperparams):

    training:
      preference:
        enabled: true
        method: dpo
        hyperparams:
          beta: 0.1
          alpha: 0.1         # ORPO default; DPO ignores
          learning_rate: 5e-6
          num_epochs: 1
        loss_type: sigmoid
        reference: pre_adapter  # renamed from pre_dpo_adapter

Docs that never set `dpo.*` keep the default `PreferenceConfig()` — the
migrator only rewrites the block when it's present. Idempotent: a v1
doc with no `training.dpo` passes through with just the top-level dict
copied.
"""

from __future__ import annotations

from typing import Any, cast

from dlm.doc.migrations import register


@register(from_version=1)
def migrate(raw: dict[str, object]) -> dict[str, object]:
    out = dict(raw)
    training = out.get("training")
    if not isinstance(training, dict):
        return out

    training_out = dict(training)
    dpo = training_out.pop("dpo", None)
    if dpo is None:
        out["training"] = training_out
        return out
    if not isinstance(dpo, dict):
        # Malformed — leave under the new key so the Pydantic validator
        # raises a useful schema error rather than the migrator
        # silently dropping the user's data.
        training_out["preference"] = dpo
        out["training"] = training_out
        return out

    dpo_map = cast(dict[str, Any], dpo)
    preference: dict[str, object] = {"method": "dpo"}
    if "enabled" in dpo_map:
        preference["enabled"] = dpo_map["enabled"]
    hyperparams: dict[str, object] = {}
    for key in ("beta", "learning_rate", "num_epochs"):
        if key in dpo_map:
            hyperparams[key] = dpo_map[key]
    if hyperparams:
        preference["hyperparams"] = hyperparams
    if "loss_type" in dpo_map:
        preference["loss_type"] = dpo_map["loss_type"]
    if "reference" in dpo_map:
        ref = dpo_map["reference"]
        preference["reference"] = "pre_adapter" if ref == "pre_dpo_adapter" else ref

    training_out["preference"] = preference
    out["training"] = training_out
    return out
