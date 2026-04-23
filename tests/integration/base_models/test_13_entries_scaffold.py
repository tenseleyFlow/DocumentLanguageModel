"""Sprint 40 closeout mirror for the named 13-entry scaffold deliverable."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.integration.cli.test_registry_refresh_init import SPRINT40_INIT_CASES
from typer.testing import CliRunner

from dlm.cli.app import app
from dlm.doc.parser import parse_file
from dlm.doc.sections import SectionType


@pytest.mark.parametrize(("base_key", "extra_flags", "expect_image_section"), SPRINT40_INIT_CASES)
def test_init_scaffolds_for_all_thirteen_registry_refresh_entries(
    tmp_path: Path,
    base_key: str,
    extra_flags: list[str],
    expect_image_section: bool,
) -> None:
    runner = CliRunner()
    home = tmp_path / "home"
    doc = tmp_path / f"{base_key}.dlm"

    result = runner.invoke(
        app,
        [
            "--home",
            str(home),
            "init",
            str(doc),
            "--base",
            base_key,
            *extra_flags,
        ],
    )
    assert result.exit_code == 0, result.output
    parsed = parse_file(doc)
    section_types = {section.type for section in parsed.sections}
    if expect_image_section:
        assert SectionType.IMAGE in section_types
    else:
        assert SectionType.IMAGE not in section_types
