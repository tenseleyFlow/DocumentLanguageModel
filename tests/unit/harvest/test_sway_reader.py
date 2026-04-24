"""Unit tests for `dlm.harvest.sway_reader` (Sprint 33.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlm.harvest import (
    HarvestCandidate,
    MalformedSwayReportError,
    NoReferenceError,
    read_sway_report,
)


def _write(tmp_path: Path, payload: object) -> Path:
    report = tmp_path / "sway.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    return report


_PROBE_FAIL_WITH_REF = {
    "name": "fortran_subroutine_semantics",
    "kind": "section_internalization",
    "verdict": "fail",
    "score": 0.22,
    "raw": 0.22,
    "z_score": -1.7,
    "evidence": {
        "prompt": "What does SUBROUTINE DGEMM compute?",
        "reference": "A double-precision general matrix multiplication.",
        "confidence": 0.9,
    },
    "message": "adapter failed semantic recall",
    "duration_s": 0.4,
}
_PROBE_FAIL_NO_REF = {
    "name": "docstring_recall",
    "kind": "prompt_collapse",
    "verdict": "fail",
    "score": 0.1,
    "evidence": {"per_section_scores": [0.1, 0.15]},
    "message": "probe has no Q/A pair to harvest",
    "duration_s": 0.2,
}
_PROBE_PASS = {
    "name": "calibration",
    "kind": "calibration_drift",
    "verdict": "pass",
    "score": 0.95,
    "evidence": {"delta": 0.01},
    "message": "calibration healthy",
    "duration_s": 0.1,
}


def _full_report(probes: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "sway_version": "0.1.0.dev0",
        "base_model_id": "smollm2-135m",
        "adapter_id": "run_7",
        "started_at": "2026-04-21T00:00:00Z",
        "finished_at": "2026-04-21T00:05:00Z",
        "wall_seconds": 300.0,
        "probes": probes,
    }


class TestHappyPath:
    def test_single_failing_probe_lifts_cleanly(self, tmp_path: Path) -> None:
        report = _write(tmp_path, _full_report([_PROBE_FAIL_WITH_REF, _PROBE_PASS]))
        candidates = read_sway_report(report)

        assert len(candidates) == 1
        c = candidates[0]
        assert isinstance(c, HarvestCandidate)
        assert c.prompt == "What does SUBROUTINE DGEMM compute?"
        assert c.reference == "A double-precision general matrix multiplication."
        assert c.confidence == pytest.approx(0.9)
        assert c.probe_name == "fortran_subroutine_semantics"
        assert c.probe_kind == "section_internalization"
        assert c.source_adapter_version == "run_7"

    def test_empty_probes_list_yields_empty(self, tmp_path: Path) -> None:
        report = _write(tmp_path, _full_report([]))
        assert read_sway_report(report) == []

    def test_all_passing_yields_empty(self, tmp_path: Path) -> None:
        report = _write(tmp_path, _full_report([_PROBE_PASS, _PROBE_PASS]))
        assert read_sway_report(report) == []

    def test_missing_adapter_id_leaves_source_version_none(self, tmp_path: Path) -> None:
        payload = _full_report([_PROBE_FAIL_WITH_REF])
        del payload["adapter_id"]
        report = _write(tmp_path, payload)
        candidates = read_sway_report(report)
        assert len(candidates) == 1
        assert candidates[0].source_adapter_version is None

    def test_min_confidence_filters(self, tmp_path: Path) -> None:
        low_conf = {**_PROBE_FAIL_WITH_REF}
        low_conf["evidence"] = {
            "prompt": "q?",
            "reference": "a.",
            "confidence": 0.5,
        }
        report = _write(tmp_path, _full_report([low_conf]))
        assert read_sway_report(report, min_confidence=0.8) == []
        assert len(read_sway_report(report, min_confidence=0.4)) == 1

    def test_missing_confidence_defaults_to_one(self, tmp_path: Path) -> None:
        no_conf = {**_PROBE_FAIL_WITH_REF}
        no_conf["evidence"] = {"prompt": "q?", "reference": "a."}
        report = _write(tmp_path, _full_report([no_conf]))
        candidates = read_sway_report(report)
        assert len(candidates) == 1
        assert candidates[0].confidence == 1.0

    def test_invalid_confidence_defaults_to_one(self, tmp_path: Path) -> None:
        broken_conf = {**_PROBE_FAIL_WITH_REF}
        broken_conf["evidence"] = {
            "prompt": "q?",
            "reference": "a.",
            "confidence": {"not": "numeric"},
        }
        report = _write(tmp_path, _full_report([broken_conf]))
        candidates = read_sway_report(report)
        assert len(candidates) == 1
        assert candidates[0].confidence == 1.0


class TestMissingReference:
    def test_strict_raises(self, tmp_path: Path) -> None:
        report = _write(tmp_path, _full_report([_PROBE_FAIL_NO_REF]))
        with pytest.raises(NoReferenceError):
            read_sway_report(report, strict=True)

    def test_lax_skips_with_log(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        report = _write(
            tmp_path,
            _full_report([_PROBE_FAIL_NO_REF, _PROBE_FAIL_WITH_REF]),
        )
        with caplog.at_level("WARNING"):
            candidates = read_sway_report(report, strict=False)
        assert len(candidates) == 1
        assert candidates[0].probe_name == "fortran_subroutine_semantics"
        assert any("carries no reference" in rec.message for rec in caplog.records)

    def test_empty_reference_string_refused(self, tmp_path: Path) -> None:
        empty = {**_PROBE_FAIL_WITH_REF}
        empty["evidence"] = {"prompt": "q?", "reference": "   "}
        report = _write(tmp_path, _full_report([empty]))
        with pytest.raises(NoReferenceError):
            read_sway_report(report, strict=True)


class TestMalformed:
    def test_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(MalformedSwayReportError, match="cannot read"):
            read_sway_report(tmp_path / "does-not-exist.json")

    def test_not_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("this is not json {", encoding="utf-8")
        with pytest.raises(MalformedSwayReportError, match="not valid JSON"):
            read_sway_report(bad)

    def test_top_level_array_rejected(self, tmp_path: Path) -> None:
        report = _write(tmp_path, [])
        with pytest.raises(MalformedSwayReportError, match="must be a JSON object"):
            read_sway_report(report)

    def test_missing_schema_version(self, tmp_path: Path) -> None:
        report = _write(tmp_path, {"sway_version": "0.1", "probes": []})
        with pytest.raises(MalformedSwayReportError, match="schema_version"):
            read_sway_report(report)

    def test_newer_schema_refused(self, tmp_path: Path) -> None:
        payload = _full_report([])
        payload["schema_version"] = 99
        report = _write(tmp_path, payload)
        with pytest.raises(MalformedSwayReportError, match="newer than this reader"):
            read_sway_report(report)

    def test_supported_schema_pinned_to_current_sway_version(self) -> None:
        """Cross-repo bump-gate: `_SUPPORTED_SWAY_SCHEMA` must match sway's.

        Sway's shipping schema is v1. If a sway bump lands but we forget
        to bump this constant (or vice versa), any operator harvesting
        newer sway output hits `test_newer_schema_refused`'s refusal
        path. This test pins the constant itself so a casual edit to
        the reader can't silently drop the support floor — the golden
        fails and forces a code review.
        """
        from dlm.harvest.sway_reader import _SUPPORTED_SWAY_SCHEMA

        assert _SUPPORTED_SWAY_SCHEMA == 1, (
            "sway schema pin changed — verify `sway/src/dlm_sway/suite/report.py` "
            "still emits this version and bump the docs pointer in `dlm.lock`"
        )

    def test_missing_probes_array(self, tmp_path: Path) -> None:
        report = _write(tmp_path, {"schema_version": 1, "sway_version": "0.1"})
        with pytest.raises(MalformedSwayReportError, match="`probes` array"):
            read_sway_report(report)

    def test_probe_evidence_not_object(self, tmp_path: Path) -> None:
        broken = {**_PROBE_FAIL_WITH_REF}
        broken["evidence"] = "not-an-object"
        report = _write(tmp_path, _full_report([broken]))
        with pytest.raises(NoReferenceError, match="evidence is not an object"):
            read_sway_report(report)

    def test_probe_is_not_object(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        report = _write(
            tmp_path,
            _full_report([_PROBE_FAIL_WITH_REF, "garbage"]),  # type: ignore[list-item]
        )
        with caplog.at_level("WARNING"):
            candidates = read_sway_report(report)
        assert len(candidates) == 1
        assert any("not an object" in rec.message for rec in caplog.records)
