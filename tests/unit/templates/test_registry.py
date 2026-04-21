"""Registry: enumerate bundled templates + validation behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.templates import (
    Template,
    TemplateMetaError,
    TemplateNotFoundError,
    bundled_templates_dir,
    list_bundled,
    load_template,
)


def test_bundled_templates_dir_exists() -> None:
    root = bundled_templates_dir()
    assert root.exists(), f"gallery dir missing at {root}"
    assert root.is_dir()


def test_list_bundled_returns_eight_templates() -> None:
    templates = list_bundled()
    assert len(templates) >= 8, (
        f"expected at least 8 templates, got {len(templates)}: {[t.name for t in templates]}"
    )
    names = {t.name for t in templates}
    required = {
        "coding-tutor",
        "domain-kb",
        "writing-partner",
        "personal-assistant",
        "changelog",
        "regex-buddy",
        "shell-one-liner",
        "meeting-notes-summarizer",
    }
    missing = required - names
    assert not missing, f"missing expected templates: {missing}"


def test_list_bundled_is_sorted() -> None:
    names = [t.name for t in list_bundled()]
    assert names == sorted(names)


def test_load_template_returns_matching_pair() -> None:
    t = load_template("coding-tutor")
    assert isinstance(t, Template)
    assert t.name == "coding-tutor"
    assert t.meta.name == "coding-tutor"
    assert t.meta.recommended_base == "qwen2.5-coder-1.5b"
    assert "decorators" in t.dlm_text.lower() or "decorator" in t.dlm_text.lower()


def test_load_template_unknown_raises() -> None:
    with pytest.raises(TemplateNotFoundError):
        load_template("does-not-exist")


def test_registry_drops_template_missing_sidecar(tmp_path: Path) -> None:
    (tmp_path / "orphan.dlm").write_text("---\ndlm_id: 01AAAA\n---\n# body\n")
    # No `.meta.yaml` — list_bundled() should log + skip, not raise.
    assert list_bundled(gallery_dir=tmp_path) == []


def test_registry_drops_template_with_malformed_meta(tmp_path: Path) -> None:
    (tmp_path / "broken.dlm").write_text("---\ndlm_id: 01AAAA\nbase_model: foo\n---\n# body\n")
    (tmp_path / "broken.meta.yaml").write_text("not: a: valid: yaml: mapping\n")
    assert list_bundled(gallery_dir=tmp_path) == []


def test_load_template_with_mismatched_name_raises(tmp_path: Path) -> None:
    (tmp_path / "fine.dlm").write_text("---\ndlm_id: 01AAAA\nbase_model: foo\n---\n# body\n")
    # meta.name doesn't match the filename stem.
    (tmp_path / "fine.meta.yaml").write_text(
        "name: different\ntitle: X\nrecommended_base: qwen2.5-1.5b\nsummary: hi\n"
    )
    with pytest.raises(TemplateMetaError):
        load_template("fine", gallery_dir=tmp_path)
