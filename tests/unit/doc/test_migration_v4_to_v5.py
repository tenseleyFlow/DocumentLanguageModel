"""v4 → v5 migrator: additive `training.precision` override (identity)."""

from __future__ import annotations

from typing import Any

from dlm.doc.migrations.v4 import migrate
from dlm.doc.schema import DlmFrontmatter

VALID_ULID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"


class TestIdentity:
    def test_empty_passthrough(self) -> None:
        raw: dict[str, Any] = {}
        out = migrate(raw)
        assert out == raw
        assert out is not raw

    def test_v4_training_block_preserved(self) -> None:
        raw: dict[str, Any] = {
            "training": {
                "lora_r": 16,
                "adapters": {
                    "knowledge": {"adapter": "lora", "lora_r": 8},
                },
            },
        }
        out = migrate(raw)
        assert out == raw


class TestValidatesAsV5:
    def test_migrated_doc_validates_without_precision(self) -> None:
        # A v4 doc with no precision override parses as v5 unchanged;
        # precision defaults to None (let the planner pick).
        raw: dict[str, Any] = {
            "dlm_id": VALID_ULID,
            "base_model": "smollm2-135m",
            "dlm_version": 4,
        }
        out = migrate(raw)
        out["dlm_version"] = 5
        fm = DlmFrontmatter.model_validate(out)
        assert fm.training.precision is None

    def test_migrated_doc_accepts_precision_override(self) -> None:
        raw: dict[str, Any] = {
            "dlm_id": VALID_ULID,
            "base_model": "smollm2-135m",
            "dlm_version": 4,
            "training": {"precision": "fp16"},
        }
        out = migrate(raw)
        out["dlm_version"] = 5
        fm = DlmFrontmatter.model_validate(out)
        assert fm.training.precision == "fp16"
