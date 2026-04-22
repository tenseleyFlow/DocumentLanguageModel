"""Shared QLoRA precision-safety fallback and gate tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlm.export.errors import PreflightError, UnsafeMergeError
from dlm.export.plan import ExportPlan
from dlm.export.precision_safety import (
    PrecisionSafetyDecision,
    require_dequantize_or_refuse,
    resolve_precision_safety,
    was_trained_with_qlora,
)


def _write_training_run(adapter_dir: Path, *, use_qlora: bool) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "training_run.json").write_text(
        json.dumps({"use_qlora": use_qlora}),
        encoding="utf-8",
    )


def _write_pinned_versions(adapter_dir: Path, *, bnb: str | None) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "pinned_versions.json").write_text(
        json.dumps({"torch": "2.4.0", "bitsandbytes": bnb}),
        encoding="utf-8",
    )


class TestWasTrainedWithQlora:
    def test_training_run_flag_wins_over_bnb_pin(self, tmp_path: Path) -> None:
        _write_training_run(tmp_path, use_qlora=False)
        _write_pinned_versions(tmp_path, bnb="0.43.1")
        assert was_trained_with_qlora(tmp_path) is False

    def test_missing_training_run_falls_back_to_pinned_versions(self, tmp_path: Path) -> None:
        _write_pinned_versions(tmp_path, bnb="0.43.1")
        assert was_trained_with_qlora(tmp_path) is True

    def test_malformed_training_run_falls_back_when_not_strict(self, tmp_path: Path) -> None:
        _write_pinned_versions(tmp_path, bnb="0.43.1")
        (tmp_path / "training_run.json").write_text("not json", encoding="utf-8")
        assert was_trained_with_qlora(tmp_path) is True

    def test_strict_mode_refuses_malformed_training_run(self, tmp_path: Path) -> None:
        _write_pinned_versions(tmp_path, bnb="0.43.1")
        (tmp_path / "training_run.json").write_text("not json", encoding="utf-8")
        with pytest.raises(PreflightError, match="training_run_json"):
            was_trained_with_qlora(tmp_path, strict_training_run=True)


class TestResolvePrecisionSafety:
    def test_unmerged_export_is_safe(self, tmp_path: Path) -> None:
        _write_training_run(tmp_path, use_qlora=True)
        decision = resolve_precision_safety(
            tmp_path,
            merged=False,
            dequantize_confirmed=False,
        )
        assert decision == PrecisionSafetyDecision(
            was_qlora=True,
            safe=True,
            reason="Separate-adapter export does not merge base weights.",
            requires_dequantize=False,
        )

    def test_merged_qlora_without_dequantize_is_unsafe(self, tmp_path: Path) -> None:
        _write_training_run(tmp_path, use_qlora=True)
        decision = resolve_precision_safety(
            tmp_path,
            merged=True,
            dequantize_confirmed=False,
        )
        assert decision.was_qlora is True
        assert decision.safe is False
        assert decision.requires_dequantize is True
        assert "Merging loses precision silently" in decision.reason

    def test_merged_qlora_with_dequantize_is_safe(self, tmp_path: Path) -> None:
        _write_training_run(tmp_path, use_qlora=True)
        decision = resolve_precision_safety(
            tmp_path,
            merged=True,
            dequantize_confirmed=True,
        )
        assert decision.safe is True
        assert decision.requires_dequantize is True
        assert "--dequantize" in decision.reason


class TestRequireDequantizeOrRefuse:
    def test_refuses_unsafe_qlora_merge(self, tmp_path: Path) -> None:
        _write_training_run(tmp_path, use_qlora=True)
        with pytest.raises(UnsafeMergeError, match="--merged --dequantize"):
            require_dequantize_or_refuse(
                ExportPlan(merged=True, dequantize_confirmed=False),
                tmp_path,
            )

    def test_returns_decision_for_safe_plain_lora_merge(self, tmp_path: Path) -> None:
        _write_training_run(tmp_path, use_qlora=False)
        decision = require_dequantize_or_refuse(
            ExportPlan(merged=True, dequantize_confirmed=False),
            tmp_path,
        )
        assert decision.safe is True
        assert decision.was_qlora is False
