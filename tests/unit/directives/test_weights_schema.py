"""`DlmTrainingConfig.weights` schema — validators + extras."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlm.directives.schema import DlmTrainingConfig


def test_weights_default_is_empty() -> None:
    cfg = DlmTrainingConfig()
    assert cfg.weights == {}


def test_weights_accepts_positive_floats() -> None:
    cfg = DlmTrainingConfig(weights={"lang": {"py": 2.0, "rs": 0.5}})
    assert cfg.weights["lang"]["py"] == 2.0
    assert cfg.weights["lang"]["rs"] == 0.5


def test_weights_accepts_zero() -> None:
    """Weight 0 drops the row — valid, not an error."""
    cfg = DlmTrainingConfig(weights={"gen": {"true": 0.0}})
    assert cfg.weights["gen"]["true"] == 0.0


def test_weights_rejects_negative() -> None:
    with pytest.raises(ValidationError) as exc:
        DlmTrainingConfig(weights={"gen": {"true": -1.0}})
    assert "must be ≥ 0" in str(exc.value)


def test_weights_rejects_negative_in_nested_entry() -> None:
    with pytest.raises(ValidationError):
        DlmTrainingConfig(
            weights={
                "lang": {"py": 2.0, "rs": -0.5},
                "gen": {"true": 1.0},
            }
        )


def test_weights_extra_fields_forbidden_at_config_level() -> None:
    """The parent model forbids unknown keys — `weights` typo rejected."""
    with pytest.raises(ValidationError):
        DlmTrainingConfig(wights={"lang": {"py": 2.0}})  # type: ignore[call-arg]


def test_weights_frozen_after_construction() -> None:
    cfg = DlmTrainingConfig(weights={"lang": {"py": 2.0}})
    with pytest.raises(ValidationError):
        cfg.weights = {}  # type: ignore[misc]
