"""v5 → v6 migrator: additive `training.sources` + `training.sources_policy`
(identity)."""

from __future__ import annotations

from typing import Any

from dlm.doc.migrations.v5 import migrate
from dlm.doc.schema import DlmFrontmatter

VALID_ULID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"


class TestIdentity:
    def test_empty_passthrough(self) -> None:
        raw: dict[str, Any] = {}
        out = migrate(raw)
        assert out == raw
        assert out is not raw

    def test_v5_training_block_preserved(self) -> None:
        raw: dict[str, Any] = {
            "training": {"lora_r": 16, "precision": "fp16"},
        }
        out = migrate(raw)
        assert out == raw


class TestValidatesAsV6:
    def test_migrated_doc_validates_without_sources(self) -> None:
        raw: dict[str, Any] = {
            "dlm_id": VALID_ULID,
            "base_model": "smollm2-135m",
            "dlm_version": 5,
        }
        out = migrate(raw)
        out["dlm_version"] = 6
        fm = DlmFrontmatter.model_validate(out)
        assert fm.training.sources is None
        assert fm.training.sources_policy == "permissive"

    def test_migrated_doc_accepts_sources(self) -> None:
        raw: dict[str, Any] = {
            "dlm_id": VALID_ULID,
            "base_model": "smollm2-135m",
            "dlm_version": 5,
            "training": {
                "sources_policy": "strict",
                "sources": [
                    {"path": "src", "include": ["**/*.py"]},
                ],
            },
        }
        out = migrate(raw)
        out["dlm_version"] = 6
        fm = DlmFrontmatter.model_validate(out)
        assert fm.training.sources is not None
        assert fm.training.sources[0].path == "src"
        assert fm.training.sources_policy == "strict"
