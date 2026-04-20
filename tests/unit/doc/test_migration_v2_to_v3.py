"""v2 → v3 migrator: additive `training.cpt` block.

v3 is additive with defaults, so the migrator is pure identity. These
tests lock that shape in: a v2 doc round-trips unchanged, and the
combined output validates under the v3 `DlmFrontmatter` with the
default `CptConfig`.
"""

from __future__ import annotations

from typing import Any

from dlm.doc.migrations.v2 import migrate
from dlm.doc.schema import CptConfig, DlmFrontmatter

VALID_ULID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"


class TestIdentityShape:
    def test_empty_dict_passthrough(self) -> None:
        raw: dict[str, Any] = {}
        out = migrate(raw)
        assert out == raw
        assert out is not raw  # copy, not alias

    def test_v2_preference_block_preserved(self) -> None:
        raw: dict[str, Any] = {
            "dlm_id": VALID_ULID,
            "base_model": "smollm2-135m",
            "training": {
                "preference": {
                    "method": "orpo",
                    "hyperparams": {"alpha": 0.15},
                },
            },
        }
        out = migrate(raw)
        assert out == raw

    def test_full_training_block_preserved(self) -> None:
        raw: dict[str, Any] = {
            "training": {
                "adapter": "lora",
                "lora_r": 16,
                "learning_rate": 1e-4,
            },
        }
        out = migrate(raw)
        assert out == raw


class TestValidatesAsV3:
    def test_migrated_doc_validates_with_default_cpt(self) -> None:
        raw: dict[str, Any] = {
            "dlm_id": VALID_ULID,
            "base_model": "smollm2-135m",
            "dlm_version": 2,
        }
        out = migrate(raw)
        # Dispatcher stamps dlm_version post-migrate; simulate that.
        out["dlm_version"] = 3
        fm = DlmFrontmatter.model_validate(out)
        assert fm.training.cpt == CptConfig()
