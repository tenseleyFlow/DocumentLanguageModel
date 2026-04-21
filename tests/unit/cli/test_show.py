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
    """`dlm init` now creates the store + manifest (audit-05 B2), so a
    truly uninitialized path means: the `.dlm` file exists but its
    store directory is absent. Simulate by writing the `.dlm` by hand
    and pointing DLM_HOME at an unrelated empty directory.
    """

    def _write_doc(self, path: Path, dlm_id: str) -> Path:
        path.write_text(
            f"---\ndlm_id: {dlm_id}\nbase_model: smollm2-135m\n---\nbody\n",
            encoding="utf-8",
        )
        return path

    def test_human_output_says_not_initialized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DLM_HOME", str(tmp_path / "fresh-home"))
        # 26-char Crockford base32 ULID (no I / L / O / U).
        doc = self._write_doc(tmp_path / "doc.dlm", "01HRSHWZ" + "0" * 18)
        runner = CliRunner()
        result = runner.invoke(app, ["show", str(doc)])
        assert result.exit_code == 0, result.output
        joined = result.output
        assert "not yet initialized" in joined or "not initialized" in joined

    def test_json_output_reports_not_initialized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DLM_HOME", str(tmp_path / "fresh-home"))
        doc = self._write_doc(tmp_path / "doc.dlm", "01HRSHWJ" + "0" * 18)
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


class TestTrainingSources:
    """`training.sources` directives surface in `dlm show --json` output."""

    def _write_doc_with_sources(self, tmp_path: Path) -> Path:
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("print(1)\n")
        (src / "b.py").write_text("print(2)\n")
        doc = tmp_path / "doc.dlm"
        doc.write_text(
            "---\n"
            "dlm_id: 01HRSHWA" + "0" * 18 + "\n"
            "dlm_version: 6\n"
            "base_model: smollm2-135m\n"
            "training:\n"
            "  sources:\n"
            "    - path: src\n"
            "      include: ['**/*.py']\n"
            "---\n"
            "body\n",
            encoding="utf-8",
        )
        return doc

    def test_json_reports_training_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DLM_HOME", str(tmp_path / "fresh-home"))
        doc = self._write_doc_with_sources(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["show", str(doc), "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "training_sources" in payload
        sources = payload["training_sources"]
        assert len(sources) == 1
        assert sources[0]["path"] == "src"
        assert sources[0]["file_count"] == 2

    def test_human_output_lists_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DLM_HOME", str(tmp_path / "fresh-home"))
        doc = self._write_doc_with_sources(tmp_path)
        # Populate store to get through to the inspection path.
        from dlm.doc.parser import parse_file
        from dlm.store.manifest import Manifest, save_manifest
        from dlm.store.paths import for_dlm

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
        assert "training sources" in result.output
        assert "src" in result.output


class TestBadInput:
    def test_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["show", str(tmp_path / "does-not-exist.dlm")])
        assert result.exit_code != 0
