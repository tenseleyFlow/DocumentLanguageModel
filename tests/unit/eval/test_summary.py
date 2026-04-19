"""TrainingSummary schema + round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dlm.eval.summary import (
    ProbeOutput,
    TrainingSummary,
    load_summary,
    save_summary,
    summary_path_for,
)


def _summary(**overrides: object) -> TrainingSummary:
    base: dict[str, object] = {
        "run_id": 1,
        "adapter_version": 1,
        "seed": 42,
        "steps": 100,
        "final_train_loss": 1.23,
        "final_val_loss": 1.45,
        "final_val_perplexity": 4.26,
        "early_stopped": False,
        "duration_seconds": 12.5,
        "determinism_class": "strict",
    }
    base.update(overrides)
    return TrainingSummary.model_validate(base)


class TestSchema:
    def test_minimal_accepted(self) -> None:
        s = TrainingSummary(run_id=1, adapter_version=1, seed=0)
        assert s.run_id == 1
        assert s.final_train_loss is None
        assert s.probes == []

    def test_run_id_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TrainingSummary(run_id=0, adapter_version=1, seed=0)

    def test_adapter_version_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TrainingSummary(run_id=1, adapter_version=0, seed=0)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            TrainingSummary.model_validate(
                {"run_id": 1, "adapter_version": 1, "seed": 0, "bonus": "nope"}
            )

    def test_frozen(self) -> None:
        s = _summary()
        with pytest.raises(ValidationError):
            s.run_id = 2  # type: ignore[misc]

    def test_probes_roundtrip(self) -> None:
        s = _summary(
            probes=[
                {"prompt": "Q?", "response": "A.", "reference": "ref", "section_id": "sid0"},
            ]
        )
        assert isinstance(s.probes[0], ProbeOutput)
        assert s.probes[0].prompt == "Q?"


class TestSaveLoad:
    def test_round_trip_via_json(self, tmp_path: Path) -> None:
        s = _summary(
            probes=[{"prompt": "Q?", "response": "A.", "reference": None, "section_id": ""}]
        )
        p = tmp_path / "summary.json"
        save_summary(p, s)
        back = load_summary(p)
        assert back == s

    def test_written_file_is_sorted_pretty_json(self, tmp_path: Path) -> None:
        s = _summary()
        p = tmp_path / "summary.json"
        save_summary(p, s)
        text = p.read_text()
        # Pretty (indented) + trailing newline.
        assert text.endswith("\n")
        assert "  " in text
        # Sorted keys mean `adapter_version` appears before `run_id`.
        data = json.loads(text)
        keys = list(data.keys())
        assert keys == sorted(keys)


class TestSummaryPathFor:
    def test_matches_log_stem(self, tmp_path: Path) -> None:
        # summary_path_for normalizes timestamps.
        p = summary_path_for(tmp_path, 7, "2026-04-18T10:15:23")
        assert p.parent == tmp_path
        assert p.name.startswith("train-000007-")
        assert p.suffix == ".json"
        assert ".summary" in p.name
