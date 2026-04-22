"""CLI behavior: `dlm templates list` + `dlm init --template`."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def test_templates_list_prints_names() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["templates", "list"])
    assert result.exit_code == 0, result.output
    assert "coding-tutor" in result.output
    assert "regex-buddy" in result.output
    assert "shell-one-liner" in result.output
    assert "meeting-notes-summarizer" in result.output


def test_templates_list_json() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["templates", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, list)
    assert len(payload) >= 8
    required_keys = {
        "name",
        "title",
        "domain_tags",
        "recommended_base",
        "expected_steps",
        "expected_duration",
        "summary",
        "sample_prompts",
    }
    for entry in payload:
        assert required_keys.issubset(entry.keys()), entry


def test_templates_list_refresh_falls_back_to_bundled() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["templates", "list", "--refresh"])
    # Refresh is currently a no-op that warns + falls back; the bundled
    # list still renders and the exit code is 0.
    assert result.exit_code == 0, result.output
    assert "not available yet" in result.output
    assert "Sprint" not in result.output
    assert "coding-tutor" in result.output


def test_init_template_writes_file(tmp_path: Path) -> None:
    runner = CliRunner()
    target = tmp_path / "mydoc.dlm"
    home = tmp_path / "dlm-home"
    result = runner.invoke(
        app,
        ["--home", str(home), "init", str(target), "--template", "shell-one-liner"],
    )
    assert result.exit_code == 0, result.output
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "qwen2.5-coder-1.5b" in text
    assert "Shell one-liner" in result.output or "shell-one-liner" in result.output


def test_init_template_unknown_name_exits_nonzero(tmp_path: Path) -> None:
    runner = CliRunner()
    target = tmp_path / "mydoc.dlm"
    home = tmp_path / "dlm-home"
    result = runner.invoke(
        app,
        ["--home", str(home), "init", str(target), "--template", "not-a-template"],
    )
    assert result.exit_code == 1
    assert not target.exists()
