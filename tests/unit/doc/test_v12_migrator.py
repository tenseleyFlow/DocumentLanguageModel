"""Named Sprint 40 closeout checks for the v12 → v13 migrator."""

from __future__ import annotations

from typing import Any

from dlm.doc.migrations.v12 import migrate
from dlm.doc.schema import DlmFrontmatter

_VALID_ULID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"


def test_v12_migrator_is_identity_for_existing_frontmatter() -> None:
    raw: dict[str, Any] = {
        "dlm_id": _VALID_ULID,
        "base_model": "smollm2-135m",
        "dlm_version": 12,
        "training": {"audio": {"auto_resample": True}, "lora_r": 16},
    }
    out = migrate(raw)
    assert out == raw
    assert out is not raw


def test_v12_migrator_output_validates_as_v13() -> None:
    raw: dict[str, Any] = {
        "dlm_id": _VALID_ULID,
        "base_model": "smollm2-135m",
        "dlm_version": 12,
        "training": {"audio": {"auto_resample": True}},
    }
    out = migrate(raw)
    out["dlm_version"] = 13
    fm = DlmFrontmatter.model_validate(out)
    assert fm.dlm_version == 13
    assert fm.training.audio.auto_resample is True
