"""Scaffold coverage for the Sprint 40 registry refresh entries we ship.

This is intentionally scoped to entries that currently exist in the
registry. Two rows in the original sprint draft still need upstream
reality work (`qwen3-1.7b-thinking`, `internvl3-2b`), so this test
guards the refresh surface we have actually landed rather than baking
stale assumptions into CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app
from dlm.doc.parser import parse_file
from dlm.doc.sections import SectionType


@pytest.mark.parametrize(
    ("base_key", "extra_flags", "expect_image_section"),
    [
        ("qwen3-1.7b", [], False),
        ("qwen3-4b", [], False),
        ("qwen3-8b", [], False),
        ("llama-3.3-8b-instruct", ["--i-accept-license"], False),
        ("phi-4-mini-reasoning", [], False),
        ("gemma-2-2b-it", ["--i-accept-license"], False),
        ("gemma-2-9b-it", ["--i-accept-license"], False),
        ("mistral-small-3.1-24b-instruct", ["--multimodal"], True),
        ("smollm3-3b", [], False),
        ("olmo-2-7b-instruct", [], False),
        ("mixtral-8x7b-instruct", [], False),
    ],
)
def test_init_scaffolds_for_landed_registry_refresh_entries(
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
    assert doc.exists()

    parsed = parse_file(doc)
    assert parsed.frontmatter.base_model == base_key
    section_types = {section.type for section in parsed.sections}
    if expect_image_section:
        assert SectionType.IMAGE in section_types
    else:
        assert SectionType.IMAGE not in section_types
