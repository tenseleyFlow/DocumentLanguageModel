"""`dlm show` CLI — pretty + --json, uninitialized-store path (Sprint 13)."""

from __future__ import annotations

import json
from datetime import datetime
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

    def test_json_exports_surface_target_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dlm.doc.parser import parse_file
        from dlm.store.manifest import ExportSummary, Manifest, save_manifest
        from dlm.store.paths import for_dlm

        home = tmp_path / "dlm-home"
        monkeypatch.setenv("DLM_HOME", str(home))

        doc = _scaffold(tmp_path)
        parsed = parse_file(doc)
        store = for_dlm(parsed.frontmatter.dlm_id)
        store.ensure_layout()
        save_manifest(
            store.manifest,
            Manifest(
                dlm_id=parsed.frontmatter.dlm_id,
                base_model="smollm2-135m",
                exports=[
                    ExportSummary(
                        exported_at=datetime(2026, 4, 23, 12, 0, 0),
                        target="ollama",
                        quant="Q4_K_M",
                        merged=False,
                        ollama_name="doc:latest",
                    ),
                    ExportSummary(
                        exported_at=datetime(2026, 4, 23, 12, 5, 0),
                        target="llama-server",
                        quant="Q4_K_M",
                        merged=False,
                    ),
                    ExportSummary(
                        exported_at=datetime(2026, 4, 23, 12, 10, 0),
                        target="mlx-serve",
                        quant="hf",
                        merged=False,
                    ),
                ],
            ),
        )

        runner = CliRunner()
        result = runner.invoke(app, ["show", str(doc), "--json"])
        assert result.exit_code == 0, result.output

        payload = json.loads(result.output)
        assert [export["target"] for export in payload["exports"]] == [
            "ollama",
            "llama-server",
            "mlx-serve",
        ]

    def test_json_surfaces_latest_preference_mining_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dlm.doc.parser import parse_file
        from dlm.metrics import MetricsRecorder, PreferenceMineEvent
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
        recorder = MetricsRecorder(store.root)
        recorder.record_preference_mine(
            PreferenceMineEvent(
                run_id=7,
                judge_name="sway",
                sample_count=4,
                mined_pairs=2,
                skipped_prompts=1,
                write_mode="applied",
            )
        )

        runner = CliRunner()
        result = runner.invoke(app, ["show", str(doc), "--json"])
        assert result.exit_code == 0, result.output

        payload = json.loads(result.output)
        pref = payload["preference_mining"]
        assert pref["last_run_id"] == 7
        assert pref["run_count"] == 1
        assert pref["event_count"] == 1
        assert pref["last_run_event_count"] == 1
        assert pref["total_mined_pairs"] == 2
        assert pref["total_skipped_prompts"] == 1
        assert pref["last_event"]["judge_name"] == "sway"
        assert pref["last_event"]["write_mode"] == "applied"
        assert payload["preference_mining_runs"] == 1
        assert payload["total_auto_mined_pairs"] == 2


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

    def test_json_reports_discovered_training_configs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sprint 30: `.dlm/training.yaml` + `.dlm/ignore` under the source
        tree surface as `discovered_training_configs` in the JSON output."""
        monkeypatch.setenv("DLM_HOME", str(tmp_path / "fresh-home"))
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("print(1)\n")
        # Drop a .dlm/training.yaml + .dlm/ignore in the source tree
        (src / ".dlm").mkdir()
        (src / ".dlm" / "training.yaml").write_text(
            "dlm_training_version: 1\nmetadata:\n  language: python\n",
            encoding="utf-8",
        )
        (src / ".dlm" / "ignore").write_text("*.log\n", encoding="utf-8")
        doc = tmp_path / "doc.dlm"
        doc.write_text(
            "---\n"
            "dlm_id: 01HRSHWE" + "0" * 18 + "\n"
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
        runner = CliRunner()
        result = runner.invoke(app, ["show", str(doc), "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "discovered_training_configs" in payload
        configs = payload["discovered_training_configs"]
        assert len(configs) == 1
        assert configs[0]["has_training_yaml"] is True
        assert configs[0]["has_ignore"] is True
        assert configs[0]["metadata"] == {"language": "python"}
        assert configs[0]["ignore_rules"] == 1

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
