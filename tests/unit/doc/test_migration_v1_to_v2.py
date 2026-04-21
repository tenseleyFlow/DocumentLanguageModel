"""v1 → v2 migrator: `training.dpo` → `training.preference`."""

from __future__ import annotations

from typing import Any

from dlm.doc.migrations.v1 import migrate


class TestNoTraining:
    def test_missing_training_block_passthrough(self) -> None:
        raw: dict[str, Any] = {"dlm_id": "x", "base_model": "y"}
        out = migrate(raw)
        assert out == raw
        assert out is not raw  # copy, not alias

    def test_training_without_dpo_passthrough(self) -> None:
        raw: dict[str, Any] = {
            "dlm_id": "x",
            "base_model": "y",
            "training": {"adapter": "lora", "lora_r": 8},
        }
        out = migrate(raw)
        assert out["training"] == {"adapter": "lora", "lora_r": 8}


class TestFullDpoBlock:
    def test_renames_and_regroups(self) -> None:
        raw: dict[str, Any] = {
            "dlm_id": "x",
            "base_model": "y",
            "training": {
                "adapter": "lora",
                "dpo": {
                    "enabled": True,
                    "beta": 0.25,
                    "loss_type": "ipo",
                    "learning_rate": 3e-6,
                    "num_epochs": 2,
                    "reference": "pre_dpo_adapter",
                },
            },
        }
        out = migrate(raw)
        training = out["training"]
        assert "dpo" not in training
        pref = training["preference"]
        assert pref == {
            "method": "dpo",
            "enabled": True,
            "hyperparams": {
                "beta": 0.25,
                "learning_rate": 3e-6,
                "num_epochs": 2,
            },
            "loss_type": "ipo",
            "reference": "pre_adapter",
        }

    def test_other_training_fields_preserved(self) -> None:
        raw: dict[str, Any] = {
            "training": {
                "adapter": "lora",
                "lora_r": 16,
                "dpo": {"enabled": True},
            },
        }
        out = migrate(raw)
        assert out["training"]["adapter"] == "lora"
        assert out["training"]["lora_r"] == 16


class TestPartialDpoBlock:
    def test_only_enabled_set(self) -> None:
        raw: dict[str, Any] = {"training": {"dpo": {"enabled": True}}}
        out = migrate(raw)
        assert out["training"]["preference"] == {"method": "dpo", "enabled": True}

    def test_only_hyperparam_fields(self) -> None:
        raw: dict[str, Any] = {"training": {"dpo": {"beta": 0.3}}}
        out = migrate(raw)
        assert out["training"]["preference"] == {
            "method": "dpo",
            "hyperparams": {"beta": 0.3},
        }


class TestReferenceRename:
    def test_pre_dpo_adapter_becomes_pre_adapter(self) -> None:
        raw: dict[str, Any] = {"training": {"dpo": {"reference": "pre_dpo_adapter"}}}
        out = migrate(raw)
        assert out["training"]["preference"]["reference"] == "pre_adapter"

    def test_base_reference_passes_through(self) -> None:
        raw: dict[str, Any] = {"training": {"dpo": {"reference": "base"}}}
        out = migrate(raw)
        assert out["training"]["preference"]["reference"] == "base"


class TestMalformedDpo:
    def test_non_dict_dpo_left_under_preference_key(self) -> None:
        """A malformed `dpo` value survives into `preference` so pydantic
        raises a useful schema error rather than the migrator silently
        dropping data."""
        raw: dict[str, Any] = {"training": {"dpo": "bogus-string"}}
        out = migrate(raw)
        assert out["training"]["preference"] == "bogus-string"


class TestIdempotentShape:
    def test_does_not_rewrite_already_v2_shape(self) -> None:
        """A doc that already has `training.preference` (no `dpo`) just
        round-trips."""
        raw: dict[str, Any] = {
            "training": {
                "preference": {
                    "method": "orpo",
                    "hyperparams": {"alpha": 0.15},
                },
            },
        }
        out = migrate(raw)
        assert out == raw
