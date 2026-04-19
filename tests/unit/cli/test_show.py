"""`dlm show` CLI — pretty + --json, uninitialized-store path (Sprint 13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app


def _scaffold(tmp_path: Path) -> Path:
    """Run `dlm init` to produce a valid .dlm file the show cmd can read."""
    doc = tmp_path / "doc.dlm"
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(doc), "--base", "smollm2-135m"])
    assert result.exit_code == 0, result.output
    return doc


class TestUninitializedStore:
    def test_human_output_says_not_initialized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DLM_HOME", str(tmp_path / "dlm-home"))
        doc = _scaffold(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["show", str(doc)])
        assert result.exit_code == 0, result.output
        joined = result.output
        # The human output mentions "not yet initialized" for absent stores.
        assert "not yet initialized" in joined or "not initialized" in joined

    def test_json_output_reports_not_initialized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DLM_HOME", str(tmp_path / "dlm-home"))
        doc = _scaffold(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["show", str(doc), "--json"])
        assert result.exit_code == 0, result.output

        parsed = json.loads(result.output)
        assert parsed["store_initialized"] is False
        assert parsed["base_model"] == "smollm2-135m"
        assert "dlm_id" in parsed


class TestInitializedStore:
    def test_human_output_renders_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Create a store manifest manually and ensure show renders it."""
        from dlm.doc.parser import parse_file
        from dlm.store.manifest import Manifest, save_manifest
        from dlm.store.paths import for_dlm

        home = tmp_path / "dlm-home"
        monkeypatch.setenv("DLM_HOME", str(home))

        doc = _scaffold(tmp_path)
        parsed = parse_file(doc)
        store = for_dlm(parsed.frontmatter.dlm_id)
        store.ensure_layout()
        save_manifest(
            store.manifest,
            Manifest(dlm_id=parsed.frontmatter.dlm_id, base_model="smollm2-135m"),
        )

        runner = CliRunner()
        result = runner.invoke(app, ["show", str(doc)])
        assert result.exit_code == 0, result.output
        joined = result.output
        assert parsed.frontmatter.dlm_id in joined
        assert "smollm2-135m" in joined
        assert "training runs" in joined

    def test_json_schema_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """JSON contract: documented keys present; gate against accidental drift."""
        from dlm.doc.parser import parse_file
        from dlm.store.manifest import Manifest, save_manifest
        from dlm.store.paths import for_dlm

        home = tmp_path / "dlm-home"
        monkeypatch.setenv("DLM_HOME", str(home))

        doc = _scaffold(tmp_path)
        parsed = parse_file(doc)
        store = for_dlm(parsed.frontmatter.dlm_id)
        store.ensure_layout()
        save_manifest(
            store.manifest,
            Manifest(dlm_id=parsed.frontmatter.dlm_id, base_model="smollm2-135m"),
        )

        runner = CliRunner()
        result = runner.invoke(app, ["show", str(doc), "--json"])
        assert result.exit_code == 0, result.output

        payload = json.loads(result.output)
        expected_keys = {
            "dlm_id",
            "path",
            "base_model",
            "base_model_revision",
            "adapter_version",
            "training_runs",
            "last_trained_at",
            "has_adapter_current",
            "replay_size_bytes",
            "total_size_bytes",
            "source_path",
            "orphaned",
            "exports",
            "content_hashes",
            "pinned_versions",
        }
        assert expected_keys.issubset(payload.keys())


class TestBadInput:
    def test_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["show", str(tmp_path / "does-not-exist.dlm")])
        assert result.exit_code != 0
