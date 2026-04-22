"""v12 → v13 migrator: identity bump for the registry refresh."""

from __future__ import annotations

from typing import Any

from dlm.doc.migrations.v12 import migrate
from dlm.doc.schema import DlmFrontmatter

VALID_ULID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"


class TestIdentity:
    def test_empty_passthrough(self) -> None:
        raw: dict[str, Any] = {}
        out = migrate(raw)
        assert out == raw
        assert out is not raw

    def test_v12_training_block_preserved(self) -> None:
        raw: dict[str, Any] = {
            "training": {"audio": {"auto_resample": True}, "lora_r": 16},
        }
        out = migrate(raw)
        assert out == raw


class TestValidatesAsV13:
    def test_migrated_v12_doc_validates_as_current(self) -> None:
        raw: dict[str, Any] = {
            "dlm_id": VALID_ULID,
            "base_model": "smollm2-135m",
            "dlm_version": 12,
            "training": {"audio": {"auto_resample": True}},
        }
        out = migrate(raw)
        out["dlm_version"] = 13
        fm = DlmFrontmatter.model_validate(out)
        assert fm.dlm_version == 13
        assert fm.training.audio.auto_resample is True
