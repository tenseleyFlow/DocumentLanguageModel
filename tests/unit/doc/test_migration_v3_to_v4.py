"""v3 → v4 migrator: additive `training.adapters` block (identity)."""

from __future__ import annotations

from typing import Any

from dlm.doc.migrations.v3 import migrate
from dlm.doc.schema import DlmFrontmatter

VALID_ULID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"


class TestIdentity:
    def test_empty_passthrough(self) -> None:
        raw: dict[str, Any] = {}
        out = migrate(raw)
        assert out == raw
        assert out is not raw

    def test_v3_cpt_block_preserved(self) -> None:
        raw: dict[str, Any] = {
            "training": {
                "cpt": {"schedule": "dapt", "embed_warmup_steps": 100},
            },
        }
        out = migrate(raw)
        assert out == raw

    def test_v3_preference_block_preserved(self) -> None:
        raw: dict[str, Any] = {
            "training": {
                "preference": {
                    "method": "orpo",
                    "hyperparams": {"alpha": 0.2},
                },
            },
        }
        out = migrate(raw)
        assert out == raw


class TestValidatesAsV4:
    def test_migrated_doc_validates_without_adapters_block(self) -> None:
        raw: dict[str, Any] = {
            "dlm_id": VALID_ULID,
            "base_model": "smollm2-135m",
            "dlm_version": 3,
        }
        out = migrate(raw)
        out["dlm_version"] = 4
        fm = DlmFrontmatter.model_validate(out)
        assert fm.training.adapters is None
