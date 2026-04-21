"""Exceptions for the control-vector pipeline."""

from __future__ import annotations


class ControlError(Exception):
    """Base for every failure inside `dlm.control`."""


class ControlExtractError(ControlError):
    """Raised when PCA-over-residuals can't produce a stable direction.

    Covers degenerate inputs (zero variance, NaN hidden states,
    mismatched chosen/rejected shapes). Caller shows the message to
    the user; no recovery path.
    """


class ControlApplyError(ControlError):
    """Raised when a control vector can't be attached to a model.

    Shape mismatch (vector dim != model hidden dim), invalid layer
    index, or an already-active hook on the same layer.
    """


class ControlPolicyRefusal(ControlError):  # noqa: N818
    """Refused to extract a control from a `policy: safety` preference.

    Documents can't self-extract vectors that would undo safety
    training — a control vector over "safe vs unsafe" chosen/rejected
    pairs is, by construction, a steering direction that erodes the
    safety behavior when applied at negative strength. We refuse at
    extraction time, not application time, so the artifact never
    reaches disk.
    """
