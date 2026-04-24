"""JSONL step logger — banner/step/event lines + atomic flush."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from dlm.train.logger import Banner, StepLogger, log_path_for


class TestContextManager:
    def test_path_property_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        log = StepLogger(path)
        assert log.path == path

    def test_outside_context_raises(self, tmp_path: Path) -> None:
        log = StepLogger(tmp_path / "x.jsonl")
        with pytest.raises(RuntimeError, match="not open"):
            log.log_step(step=1, loss=1.0, lr=1e-5)

    def test_file_closed_on_exit(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        with StepLogger(path) as log:
            log.log_step(step=1, loss=1.0, lr=1e-5)
        # Re-open should work cleanly (previous handle closed).
        with StepLogger(path) as log:
            log.log_step(step=2, loss=0.5, lr=1e-5)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2


class TestBanner:
    def test_banner_is_first_line(self, tmp_path: Path) -> None:
        p = tmp_path / "run.jsonl"
        with StepLogger(p) as log:
            log.write_banner(Banner(run_id=3, seed=42, determinism_class="strict"))
            log.log_step(step=1, loss=1.0, lr=1e-5)
        first = json.loads(p.read_text().splitlines()[0])
        assert first["type"] == "banner"
        assert first["run_id"] == 3

    def test_banner_carries_pinned_versions(self, tmp_path: Path) -> None:
        p = tmp_path / "run.jsonl"
        with StepLogger(p) as log:
            log.write_banner(
                Banner(
                    run_id=1,
                    seed=0,
                    pinned_versions=(("torch", "2.4.0"), ("trl", "0.12.0")),
                )
            )
        parsed = json.loads(p.read_text().strip())
        assert parsed["pinned_versions"] == [["torch", "2.4.0"], ["trl", "0.12.0"]]


class TestStepLogging:
    def test_step_has_all_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "run.jsonl"
        with StepLogger(p) as log:
            log.log_step(step=5, loss=2.5, lr=1e-4, grad_norm=0.8, tokens_per_sec=1000.0)
        parsed = json.loads(p.read_text().strip())
        assert parsed["step"] == 5
        assert parsed["loss"] == 2.5
        assert parsed["lr"] == 1e-4
        assert parsed["grad_norm"] == 0.8
        assert parsed["tokens_per_sec"] == 1000.0
        assert parsed["type"] == "step"
        assert parsed["timestamp"]  # non-empty

    def test_optional_fields_null(self, tmp_path: Path) -> None:
        p = tmp_path / "run.jsonl"
        with StepLogger(p) as log:
            log.log_step(step=1, loss=1.0, lr=1e-5)
        parsed = json.loads(p.read_text().strip())
        assert parsed["grad_norm"] is None
        assert parsed["tokens_per_sec"] is None
        assert parsed["val_loss"] is None


class TestEventLogging:
    def test_custom_event_type(self, tmp_path: Path) -> None:
        p = tmp_path / "run.jsonl"
        with StepLogger(p) as log:
            log.log_event("oom", step=3, peak_bytes=1_000_000_000)
        parsed = json.loads(p.read_text().strip())
        assert parsed["type"] == "oom"
        assert parsed["step"] == 3
        assert parsed["peak_bytes"] == 1_000_000_000
        assert parsed["timestamp"]

    def test_event_survives_eval_completed(self, tmp_path: Path) -> None:
        """Smoke: evaluator-style event ingests cleanly."""
        p = tmp_path / "run.jsonl"
        with StepLogger(p) as log:
            log.log_event("eval", step=10, val_loss=1.5, val_ppl=4.5)
        parsed = json.loads(p.read_text().strip())
        assert parsed["val_ppl"] == 4.5

    def test_dataclass_fields_are_sanitized(self, tmp_path: Path) -> None:
        @dataclass
        class _Payload:
            step: int
            note: str

        p = tmp_path / "run.jsonl"
        with StepLogger(p) as log:
            log.log_event("custom", payload=_Payload(step=3, note="ok"))
        parsed = json.loads(p.read_text().strip())
        assert parsed["payload"] == {"step": 3, "note": "ok"}


class TestLogPath:
    def test_shape(self, tmp_path: Path) -> None:
        p = log_path_for(tmp_path, run_id=7)
        assert p.parent == tmp_path
        assert p.name.startswith("train-000007-")
        assert p.suffix == ".jsonl"
