"""apply_template: ULID rotation + overwrite refusal + frontmatter preserved."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.doc.parser import parse_file
from dlm.templates import TemplateApplyError, TemplateNotFoundError, apply_template


def test_apply_template_writes_target(tmp_path: Path) -> None:
    target = tmp_path / "out.dlm"
    result = apply_template("coding-tutor", target)
    assert target.exists()
    assert result.template.name == "coding-tutor"
    assert result.dlm_id  # non-empty


def test_apply_template_rotates_dlm_id(tmp_path: Path) -> None:
    target = tmp_path / "out.dlm"
    result = apply_template("coding-tutor", target)
    parsed = parse_file(target)
    # ULID must be the fresh one apply_template returned, not the
    # bundled template's canonical ID.
    assert parsed.frontmatter.dlm_id == result.dlm_id
    # And sanity-check that we didn't just echo the bundled ID.
    bundled = parse_file(result.template.source_path)
    assert parsed.frontmatter.dlm_id != bundled.frontmatter.dlm_id


def test_apply_template_preserves_base_model(tmp_path: Path) -> None:
    target = tmp_path / "out.dlm"
    result = apply_template("shell-one-liner", target)
    parsed = parse_file(target)
    assert parsed.frontmatter.base_model == result.template.meta.recommended_base


def test_apply_template_refuses_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "out.dlm"
    target.write_text("already here", encoding="utf-8")
    with pytest.raises(TemplateApplyError):
        apply_template("coding-tutor", target)


def test_apply_template_force_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "out.dlm"
    target.write_text("already here", encoding="utf-8")
    apply_template("coding-tutor", target, force=True)
    # File got replaced — no longer the placeholder.
    assert target.read_text(encoding="utf-8") != "already here"


def test_apply_template_unknown_name_raises(tmp_path: Path) -> None:
    target = tmp_path / "out.dlm"
    with pytest.raises(TemplateNotFoundError):
        apply_template("nonexistent-template", target)
    # And doesn't leave a half-written file behind.
    assert not target.exists()
