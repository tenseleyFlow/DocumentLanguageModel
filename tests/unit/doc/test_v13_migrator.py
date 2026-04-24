"""v13 → v14 migrator: identity bump for auto-mined preference metadata."""

from __future__ import annotations

from typing import Any

from dlm.doc.migrations.v13 import migrate
from dlm.doc.schema import DlmFrontmatter

VALID_ULID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"


def test_v13_migrator_is_identity_for_existing_frontmatter() -> None:
    raw: dict[str, Any] = {
        "dlm_id": VALID_ULID,
        "base_model": "smollm2-135m",
        "dlm_version": 13,
        "training": {"audio": {"auto_resample": True}},
    }
    out = migrate(raw)
    assert out == raw
    assert out is not raw


def test_v13_migrator_output_validates_as_v14() -> None:
    raw: dict[str, Any] = {
        "dlm_id": VALID_ULID,
        "base_model": "smollm2-135m",
        "dlm_version": 13,
        "training": {"audio": {"auto_resample": True}},
    }
    out = migrate(raw)
    out["dlm_version"] = 14
    fm = DlmFrontmatter.model_validate(out)
    assert fm.dlm_version == 14
    assert fm.training.audio.auto_resample is True
