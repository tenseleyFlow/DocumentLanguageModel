"""Control vectors as first-class artifacts.

A control vector is a one-shot steering direction extracted from
`::preference::` section hidden states. The extraction is
gradient-free (PCA over chosen − rejected residuals) and takes
seconds-to-minutes rather than the LoRA retraining path's hours.
Applied at inference via a `forward_pre_hook` on the residual
stream, it nudges generation toward the chosen distribution
without modifying weights.

Package layout mirrors `dlm.directives`: `extract` produces the
vector, `apply` attaches it, `errors` holds the exception
hierarchy. Store layout lives under
`~/.dlm/store/<dlm_id>/controls/<name>.safetensors` +
`<name>.meta.json`.
"""

from __future__ import annotations

from dlm.control.apply import apply_control
from dlm.control.errors import (
    ControlApplyError,
    ControlError,
    ControlExtractError,
    ControlPolicyRefusal,
)
from dlm.control.extract import (
    ControlVector,
    extract_control_vector,
    refuse_if_policy_safety,
)

__all__ = [
    "ControlApplyError",
    "ControlError",
    "ControlExtractError",
    "ControlPolicyRefusal",
    "ControlVector",
    "apply_control",
    "extract_control_vector",
    "refuse_if_policy_safety",
]
