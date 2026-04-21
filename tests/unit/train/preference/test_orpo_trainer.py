"""ORPO config kwarg mapping — `PreferenceConfig` → TRL `ORPOConfig`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dlm.doc.schema import PreferenceConfig, PreferenceHyperparams
from dlm.train.preference.orpo_trainer import build_orpo_config_kwargs


def _plan(
    *,
    precision: str = "fp16",
    micro_batch_size: int = 4,
    grad_accum: int = 2,
) -> SimpleNamespace:
    return SimpleNamespace(
        precision=precision,
        attn_implementation="sdpa",
        use_qlora=False,
        quant_compute_dtype=None,
        micro_batch_size=micro_batch_size,
        grad_accum=grad_accum,
        effective_batch_size=micro_batch_size * grad_accum,
        gradient_checkpointing=False,
        est_peak_vram_gb=1.0,
        est_step_seconds=0.1,
        reason="test",
        to_dict=lambda: {"precision": precision},
    )


class TestCoreMapping:
    def test_alpha_routes_to_trl_beta(self, tmp_path: Path) -> None:
        cfg = PreferenceConfig(
            enabled=True,
            method="orpo",
            hyperparams=PreferenceHyperparams(alpha=0.15, learning_rate=3e-6, num_epochs=2),
        )
        kwargs = build_orpo_config_kwargs(
            cfg, _plan(), output_dir=tmp_path, max_length=1024, seed=7
        )
        # TRL uses `beta` as its kwarg; our alpha flows through under
        # that name, independent of DPO's beta.
        assert kwargs["beta"] == 0.15
        assert kwargs["learning_rate"] == 3e-6
        assert kwargs["num_train_epochs"] == 2
        assert kwargs["seed"] == 7
        assert kwargs["data_seed"] == 7

    def test_batch_sizes_from_plan(self, tmp_path: Path) -> None:
        kwargs = build_orpo_config_kwargs(
            PreferenceConfig(enabled=True, method="orpo"),
            _plan(micro_batch_size=8, grad_accum=3),
            output_dir=tmp_path,
            max_length=2048,
            seed=42,
        )
        assert kwargs["per_device_train_batch_size"] == 8
        assert kwargs["gradient_accumulation_steps"] == 3

    def test_max_length_passthrough(self, tmp_path: Path) -> None:
        kwargs = build_orpo_config_kwargs(
            PreferenceConfig(enabled=True, method="orpo"),
            _plan(),
            output_dir=tmp_path,
            max_length=512,
            seed=0,
        )
        assert kwargs["max_length"] == 512
        # ORPO (like modern DPO) has no separate max_prompt_length.
        assert "max_prompt_length" not in kwargs


class TestSuppressedSurfaces:
    def test_report_to_empty(self, tmp_path: Path) -> None:
        kwargs = build_orpo_config_kwargs(
            PreferenceConfig(enabled=True, method="orpo"),
            _plan(),
            output_dir=tmp_path,
            max_length=1024,
            seed=0,
        )
        assert kwargs["report_to"] == []

    def test_save_strategy_no(self, tmp_path: Path) -> None:
        kwargs = build_orpo_config_kwargs(
            PreferenceConfig(enabled=True, method="orpo"),
            _plan(),
            output_dir=tmp_path,
            max_length=1024,
            seed=0,
        )
        assert kwargs["save_strategy"] == "no"


class TestMaxStepsOptional:
    def test_max_steps_absent_by_default(self, tmp_path: Path) -> None:
        kwargs = build_orpo_config_kwargs(
            PreferenceConfig(enabled=True, method="orpo"),
            _plan(),
            output_dir=tmp_path,
            max_length=1024,
            seed=0,
        )
        assert "max_steps" not in kwargs

    def test_max_steps_included(self, tmp_path: Path) -> None:
        kwargs = build_orpo_config_kwargs(
            PreferenceConfig(enabled=True, method="orpo"),
            _plan(),
            output_dir=tmp_path,
            max_length=1024,
            seed=0,
            max_steps=30,
        )
        assert kwargs["max_steps"] == 30
