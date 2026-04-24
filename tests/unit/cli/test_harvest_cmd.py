"""CLI tests for `dlm harvest` (Sprint 33.4)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app

_FRONTMATTER = (
    "---\ndlm_id: 01KPQ9X1000000000000000000\ndlm_version: 7\nbase_model: smollm2-135m\n---\n"
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _write_dlm(path: Path, body: str = "") -> None:
    path.write_text(_FRONTMATTER + body, encoding="utf-8")


def _write_sway(path: Path, probes: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sway_version": "0.1.0.dev0",
                "base_model_id": "smollm2-135m",
                "adapter_id": "run_7",
                "probes": probes,
            }
        ),
        encoding="utf-8",
    )


def _normalized_output(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(_ANSI_RE.sub("", text).split())


_FAIL_WITH_REF = {
    "name": "dgemm_semantics",
    "kind": "section_internalization",
    "verdict": "fail",
    "score": 0.2,
    "evidence": {
        "prompt": "What does DGEMM compute?",
        "reference": "A double-precision general matrix multiply.",
        "confidence": 0.9,
    },
    "message": "adapter failed recall",
    "duration_s": 0.1,
}


class TestHarvestCmd:
    def test_dry_run_default_no_write(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        sway = tmp_path / "sway.json"
        _write_dlm(doc, "prior body\n")
        _write_sway(sway, [_FAIL_WITH_REF])
        before = doc.read_text(encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            app, ["--home", str(tmp_path), "harvest", str(doc), "--sway-json", str(sway)]
        )
        assert result.exit_code == 0, result.output
        assert "harvest plan" in result.output
        assert "DGEMM" in result.output
        assert "dry-run" in result.output
        # file untouched
        assert doc.read_text(encoding="utf-8") == before

    def test_apply_writes_sections(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        sway = tmp_path / "sway.json"
        _write_dlm(doc, "prior body\n")
        _write_sway(sway, [_FAIL_WITH_REF])

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "harvest",
                str(doc),
                "--sway-json",
                str(sway),
                "--apply",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "wrote 1 section" in result.output
        # Re-parse — section with auto_harvest lives on disk.
        from dlm.doc.parser import parse_file

        reloaded = parse_file(doc)
        harvested = [s for s in reloaded.sections if s.auto_harvest]
        assert len(harvested) == 1
        assert "DGEMM" in harvested[0].content

    def test_no_candidates_exit_2(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        sway = tmp_path / "sway.json"
        _write_dlm(doc)
        _write_sway(sway, [])  # zero probes

        runner = CliRunner()
        result = runner.invoke(
            app, ["--home", str(tmp_path), "harvest", str(doc), "--sway-json", str(sway)]
        )
        assert result.exit_code == 2, result.output
        assert "no candidates" in result.output

    def test_malformed_sway_exit_1(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        sway = tmp_path / "sway.json"
        _write_dlm(doc)
        sway.write_text("not json {", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            app, ["--home", str(tmp_path), "harvest", str(doc), "--sway-json", str(sway)]
        )
        assert result.exit_code == 1, result.output
        assert "not valid JSON" in _normalized_output(result)

    def test_missing_reference_strict_exit_1(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        sway = tmp_path / "sway.json"
        _write_dlm(doc)
        no_ref = {**_FAIL_WITH_REF}
        no_ref["evidence"] = {}
        _write_sway(sway, [no_ref])

        runner = CliRunner()
        result = runner.invoke(
            app, ["--home", str(tmp_path), "harvest", str(doc), "--sway-json", str(sway)]
        )
        assert result.exit_code == 1, result.output
        assert "--lax" in result.output

    def test_lax_mode_skips_bad_probes(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        sway = tmp_path / "sway.json"
        _write_dlm(doc)
        no_ref = {**_FAIL_WITH_REF, "evidence": {}}
        _write_sway(sway, [no_ref, _FAIL_WITH_REF])

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "harvest",
                str(doc),
                "--sway-json",
                str(sway),
                "--lax",
            ],
        )
        # One candidate survived, dry-run exit 0
        assert result.exit_code == 0, result.output

    def test_revert_strips_auto_harvest(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        sway = tmp_path / "sway.json"
        _write_dlm(doc, "keep this\n")
        _write_sway(sway, [_FAIL_WITH_REF])
        runner = CliRunner()
        # First apply
        runner.invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "harvest",
                str(doc),
                "--sway-json",
                str(sway),
                "--apply",
            ],
        )
        # Then revert
        result = runner.invoke(app, ["--home", str(tmp_path), "harvest", str(doc), "--revert"])
        assert result.exit_code == 0, result.output
        assert "stripped 1" in result.output
        assert "all harvest runs" in _normalized_output(result)

        from dlm.doc.parser import parse_file

        reloaded = parse_file(doc)
        assert not any(s.auto_harvest for s in reloaded.sections)
        assert any("keep this" in s.content for s in reloaded.sections)

    def test_revert_and_sway_mutually_exclusive(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        sway = tmp_path / "sway.json"
        _write_dlm(doc)
        _write_sway(sway, [])
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "harvest",
                str(doc),
                "--sway-json",
                str(sway),
                "--revert",
            ],
        )
        assert result.exit_code == 1, result.output
        assert "mutually exclusive" in result.output

    def test_missing_sway_json_refused(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        _write_dlm(doc)
        runner = CliRunner()
        result = runner.invoke(app, ["--home", str(tmp_path), "harvest", str(doc)])
        assert result.exit_code == 1, result.output
        assert "--sway-json is required" in result.output

    def test_custom_tag_flows_to_harvest_source(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        sway = tmp_path / "sway.json"
        _write_dlm(doc)
        _write_sway(sway, [_FAIL_WITH_REF])

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "harvest",
                str(doc),
                "--sway-json",
                str(sway),
                "--tag",
                "nightly-ci",
                "--apply",
            ],
        )
        assert result.exit_code == 0, result.output

        from dlm.doc.parser import parse_file

        reloaded = parse_file(doc)
        harvested = [s for s in reloaded.sections if s.auto_harvest]
        assert len(harvested) == 1
        assert harvested[0].harvest_source == "nightly-ci/dgemm_semantics"
