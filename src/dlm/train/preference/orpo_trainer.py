"""Configure `trl.ORPOTrainer` from our `PreferenceConfig` + hardware plan.

Same split as `dpo_trainer.py`:

1. `build_orpo_config_kwargs` — pure mapping from `PreferenceConfig` +
   `TrainingPlan` to the keyword args TRL's `ORPOConfig` wants.
2. `build_orpo_trainer` — heavy: instantiates TRL's ORPOTrainer.
   `# pragma: no cover`; the slow integration suite drives it.

ORPO differs from DPO in two key ways: no reference model (the SFT
loss inside ORPO's objective anchors the policy) and a single alpha
hyperparameter instead of DPO's beta. TRL's `ORPOConfig` exposes
`beta` as the alpha knob — we translate at the kwarg boundary so
`PreferenceHyperparams.alpha` flows through under its TRL name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

    from dlm.doc.schema import PreferenceConfig
    from dlm.hardware.plan import TrainingPlan


def build_orpo_config_kwargs(
    pref_cfg: PreferenceConfig,
    plan: TrainingPlan,
    *,
    output_dir: Path,
    max_length: int,
    seed: int,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Pure mapping `PreferenceConfig` → TRL `ORPOConfig(**kwargs)`.

    `hyperparams.alpha` maps to TRL's `beta` kwarg; that's TRL's name
    for the odds-ratio weight, not ours. `loss_type` / `reference`
    fields are DPO-only and ignored here.

    The orchestrator owns checkpoint commit, so `save_strategy="no"`
    stays on; telemetry (wandb, tensorboard) stays dark.
    """
    hp = pref_cfg.hyperparams
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": hp.learning_rate,
        "num_train_epochs": hp.num_epochs,
        "beta": hp.alpha,
        "per_device_train_batch_size": plan.micro_batch_size,
        "gradient_accumulation_steps": plan.grad_accum,
        "max_length": max_length,
        "seed": seed,
        "data_seed": seed,
        "bf16": plan.precision == "bf16",
        "fp16": plan.precision == "fp16",
        "gradient_checkpointing": plan.gradient_checkpointing,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
    }
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return kwargs


def build_orpo_trainer(  # pragma: no cover
    *,
    policy_model: Any,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    pref_cfg: PreferenceConfig,
    plan: TrainingPlan,
    output_dir: Path,
    max_length: int,
    seed: int,
    max_steps: int | None = None,
) -> Any:
    """Instantiate `trl.ORPOTrainer` with our config.

    Heavy import; covered by the slow integration test. ORPO has no
    reference model — the policy carries both the SFT and preference
    objectives under one loss.
    """
    from trl import ORPOConfig, ORPOTrainer  # type: ignore[attr-defined]

    kwargs = build_orpo_config_kwargs(
        pref_cfg,
        plan,
        output_dir=output_dir,
        max_length=max_length,
        seed=seed,
        max_steps=max_steps,
    )
    trl_config = ORPOConfig(**kwargs)
    return ORPOTrainer(
        model=policy_model,
        args=trl_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
