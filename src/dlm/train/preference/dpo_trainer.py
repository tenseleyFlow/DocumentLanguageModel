"""Configure `trl.DPOTrainer` from our `DpoConfig` + hardware plan.

Split into three layers so the first two are unit-testable without HF
imports:

1. `build_dpo_config_kwargs` — pure mapping from `DpoConfig` +
   `TrainingPlan` to the keyword args TRL's `DPOConfig` wants.
2. `load_reference_model` — heavy: materializes the frozen reference
   PEFT model for the `pre_dpo_adapter` mode. `# pragma: no cover` —
   exercised by the slow integration suite.
3. `build_dpo_trainer` — heavy: instantiates TRL's DPOTrainer with the
   policy + reference + dataset + kwargs.

The slow integration test at `tests/integration/train/preference/`
drives (2) and (3); unit coverage here lives on (1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dlm.train.preference.errors import DpoReferenceLoadError

if TYPE_CHECKING:
    from pathlib import Path

    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

    from dlm.base_models import BaseModelSpec
    from dlm.doc.schema import DpoConfig
    from dlm.hardware.plan import TrainingPlan


def build_dpo_config_kwargs(
    dpo_cfg: DpoConfig,
    plan: TrainingPlan,
    *,
    output_dir: Path,
    max_length: int,
    seed: int,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Pure mapping from our config to the TRL `DPOConfig(**kwargs)`
    signature. Intentionally narrow — we only surface the knobs our
    `DpoConfig` exposes plus plan-derived batch sizing. Callers that
    need more (e.g. custom logging cadence) can post-process the dict.

    `max_prompt_length` is set to half of `max_length`, which matches
    the TRL default and keeps prompt+completion within the model's
    context window for our typical base registry (2k-8k tokens).
    """
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": dpo_cfg.learning_rate,
        "num_train_epochs": dpo_cfg.num_epochs,
        "beta": dpo_cfg.beta,
        "loss_type": dpo_cfg.loss_type,
        "per_device_train_batch_size": plan.micro_batch_size,
        "gradient_accumulation_steps": plan.grad_accum,
        "max_length": max_length,
        "max_prompt_length": max_length // 2,
        "seed": seed,
        "data_seed": seed,
        "bf16": plan.precision == "bf16",
        "fp16": plan.precision == "fp16",
        "gradient_checkpointing": plan.gradient_checkpointing,
        "logging_steps": 1,
        # Orchestrator handles the two-phase adapter commit itself, so
        # we never want TRL writing mid-run checkpoints.
        "save_strategy": "no",
        # Keep telemetry surfaces off; wandb / tensorboard stay dark.
        "report_to": [],
    }
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return kwargs


def load_reference_model(  # pragma: no cover
    spec: BaseModelSpec,
    plan: TrainingPlan,
    *,
    adapter_path: Path | None,
    mode: str,
) -> Any:
    """Materialize the frozen reference model for DPO.

    - `mode="base"` — load the bare base at `plan.load_dtype`; no
      adapter attached. DPO learns to move the policy adapter away
      from the unmodified base.
    - `mode="pre_dpo_adapter"` — load base, attach the SFT-trained
      adapter as `PeftModel`, call `.eval()`, disable grads. DPO
      learns to nudge relative to already-doc-trained behavior.

    Either way, the returned model has `requires_grad=False` on all
    parameters — TRL expects the reference to be frozen.
    """
    model = _load_bare_base(spec, plan)

    if mode == "base":
        _freeze(model)
        return model

    if mode == "pre_dpo_adapter":
        if adapter_path is None:
            raise DpoReferenceLoadError(
                adapter_path="<none>",
                cause="reference=pre_dpo_adapter requires a prior adapter version",
            )
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise DpoReferenceLoadError(
                adapter_path=str(adapter_path), cause=f"peft import failed: {exc}"
            ) from exc
        try:
            ref = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        except Exception as exc:
            raise DpoReferenceLoadError(
                adapter_path=str(adapter_path), cause=str(exc)
            ) from exc
        _freeze(ref)
        return ref

    raise ValueError(f"unknown DPO reference mode: {mode!r}")


def _load_bare_base(spec: BaseModelSpec, plan: TrainingPlan) -> Any:  # pragma: no cover
    """Tight copy of the SFT loader minus QLoRA + adapter attachment.
    Kept separate so the reference model is always a clean fp16/bf16
    base — a 4-bit quantized reference is a known DPO stability
    footgun."""
    from dataclasses import replace

    from dlm.train.loader import load_base_model

    ref_plan = replace(plan, use_qlora=False)
    return load_base_model(spec, ref_plan)


def _freeze(model: Any) -> None:  # pragma: no cover
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def build_dpo_trainer(  # pragma: no cover
    *,
    policy_model: Any,
    ref_model: Any,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    dpo_cfg: DpoConfig,
    plan: TrainingPlan,
    output_dir: Path,
    max_length: int,
    seed: int,
    max_steps: int | None = None,
) -> Any:
    """Instantiate `trl.DPOTrainer` with our config.

    Heavy import; covered by the slow integration test."""
    from trl import DPOConfig, DPOTrainer  # type: ignore[attr-defined]

    kwargs = build_dpo_config_kwargs(
        dpo_cfg,
        plan,
        output_dir=output_dir,
        max_length=max_length,
        seed=seed,
        max_steps=max_steps,
    )
    trl_config = DPOConfig(**kwargs)
    return DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=trl_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
