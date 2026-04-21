"""Every shipped starter template must parse cleanly.

Guards the cookbook recipes: if a schema bump lands (Sprint 12b
migrator chain), the templates need to move forward with it. This
test fails loudly if a checked-in template drifts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.doc.parser import parse_file
from dlm.templates import bundled_templates_dir


def _template_paths() -> list[Path]:
    return sorted(bundled_templates_dir().glob("*.dlm"))


def test_templates_dir_is_populated() -> None:
    # Guard against a silent deletion of the bundled gallery.
    paths = _template_paths()
    assert len(paths) >= 8, (
        f"expected at least 8 gallery templates under {bundled_templates_dir()}, got {len(paths)}"
    )


@pytest.mark.parametrize("template", _template_paths(), ids=lambda p: p.name)
def test_template_parses_and_has_sections(template: Path) -> None:
    parsed = parse_file(template)
    assert parsed.frontmatter.dlm_id, f"{template.name} missing dlm_id"
    assert parsed.frontmatter.base_model, f"{template.name} missing base_model"
    assert len(parsed.sections) >= 1, f"{template.name} has no body sections"
