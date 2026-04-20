"""Unit tests for the pure parts of `dpo_trainer`.

The heavy HF/TRL wiring (`load_reference_model`, `build_dpo_trainer`)
is `# pragma: no cover` and exercised by the slow integration suite.
This file only covers `build_dpo_config_kwargs`, which is a pure
mapping over data classes.
"""

from __future__ import annotations

from pathlib import Path

from dlm.doc.schema import DpoConfig
from dlm.hardware.plan import TrainingPlan
from dlm.train.preference.dpo_trainer import build_dpo_config_kwargs


def _plan(
    *,
    precision: str = "fp16",
    micro_batch_size: int = 4,
    grad_accum: int = 2,
    gradient_checkpointing: bool = False,
) -> TrainingPlan:
    return TrainingPlan(
        precision=precision,  # type: ignore[arg-type]
        attn_implementation="sdpa",
        use_qlora=False,
        quant_compute_dtype=None,
        micro_batch_size=micro_batch_size,
        grad_accum=grad_accum,
        effective_batch_size=micro_batch_size * grad_accum,
        gradient_checkpointing=gradient_checkpointing,
        est_peak_vram_gb=3.4,
        est_step_seconds=4.1,
        reason="test plan",
    )


class TestCoreFields:
    def test_routes_config_knobs_to_trl(self, tmp_path: Path) -> None:
        cfg = DpoConfig(
            enabled=True,
            beta=0.2,
            loss_type="ipo",
            learning_rate=3e-6,
            num_epochs=2,
        )
        kwargs = build_dpo_config_kwargs(
            cfg, _plan(), output_dir=tmp_path, max_length=1024, seed=7
        )
        assert kwargs["output_dir"] == str(tmp_path)
        assert kwargs["learning_rate"] == 3e-6
        assert kwargs["num_train_epochs"] == 2
        assert kwargs["beta"] == 0.2
        assert kwargs["loss_type"] == "ipo"
        assert kwargs["seed"] == 7
        assert kwargs["data_seed"] == 7

    def test_batch_sizes_passed_through_from_plan(self, tmp_path: Path) -> None:
        kwargs = build_dpo_config_kwargs(
            DpoConfig(enabled=True),
            _plan(micro_batch_size=8, grad_accum=3),
            output_dir=tmp_path,
            max_length=2048,
            seed=42,
        )
        assert kwargs["per_device_train_batch_size"] == 8
        assert kwargs["gradient_accumulation_steps"] == 3

    def test_max_prompt_length_is_half_of_max_length(self, tmp_path: Path) -> None:
        kwargs = build_dpo_config_kwargs(
            DpoConfig(enabled=True),
            _plan(),
            output_dir=tmp_path,
            max_length=1024,
            seed=0,
        )
        assert kwargs["max_length"] == 1024
        assert kwargs["max_prompt_length"] == 512


class TestPrecisionFlags:
    def test_bf16_sets_bf16_true_fp16_false(self, tmp_path: Path) -> None:
        kwargs = build_dpo_config_kwargs(
            DpoConfig(enabled=True),
            _plan(precision="bf16"),
            output_dir=tmp_path,
            max_length=1024,
            seed=0,
        )
        assert kwargs["bf16"] is True
        assert kwargs["fp16"] is False

    def test_fp16_sets_fp16_true_bf16_false(self, tmp_path: Path) -> None:
        kwargs = build_dpo_config_kwargs(
            DpoConfig(enabled=True),
            _plan(precision="fp16"),
            output_dir=tmp_path,
            max_length=1024,
            seed=0,
        )
        assert kwargs["bf16"] is False
        assert kwargs["fp16"] is True


class TestTelemetryAndCheckpointing:
    def test_report_to_is_empty_list(self, tmp_path: Path) -> None:
        """Keep wandb/tensorboard off."""
        kwargs = build_dpo_config_kwargs(
            DpoConfig(enabled=True),
            _plan(),
            output_dir=tmp_path,
            max_length=1024,
            seed=0,
        )
        assert kwargs["report_to"] == []

    def test_save_strategy_is_no(self, tmp_path: Path) -> None:
        """Orchestrator handles two-phase adapter commit; TRL should
        not write intermediate checkpoints."""
        kwargs = build_dpo_config_kwargs(
            DpoConfig(enabled=True),
            _plan(),
            output_dir=tmp_path,
            max_length=1024,
            seed=0,
        )
        assert kwargs["save_strategy"] == "no"

    def test_gradient_checkpointing_follows_plan(self, tmp_path: Path) -> None:
        kwargs_off = build_dpo_config_kwargs(
            DpoConfig(enabled=True),
            _plan(gradient_checkpointing=False),
            output_dir=tmp_path,
            max_length=1024,
            seed=0,
        )
        kwargs_on = build_dpo_config_kwargs(
            DpoConfig(enabled=True),
            _plan(gradient_checkpointing=True),
            output_dir=tmp_path,
            max_length=1024,
            seed=0,
        )
        assert kwargs_off["gradient_checkpointing"] is False
        assert kwargs_on["gradient_checkpointing"] is True


class TestMaxStepsOptional:
    def test_max_steps_absent_when_none(self, tmp_path: Path) -> None:
        kwargs = build_dpo_config_kwargs(
            DpoConfig(enabled=True),
            _plan(),
            output_dir=tmp_path,
            max_length=1024,
            seed=0,
            max_steps=None,
        )
        assert "max_steps" not in kwargs

    def test_max_steps_included_when_set(self, tmp_path: Path) -> None:
        kwargs = build_dpo_config_kwargs(
            DpoConfig(enabled=True),
            _plan(),
            output_dir=tmp_path,
            max_length=1024,
            seed=0,
            max_steps=50,
        )
        assert kwargs["max_steps"] == 50
