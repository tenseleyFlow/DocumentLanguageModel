"""LoRA/QLoRA adapter construction + resume.

Three entry points:

- `build_lora_config(spec, lora_r, lora_alpha, lora_dropout, tokenizer_grew)`
  returns a `peft.LoraConfig` — pure-python, testable without torch.
- `build_or_resume_adapter(model, spec, lora_r, ..., mode, resume_path)`
  wraps the loaded base model in a `PeftModel`. Two modes:
  - `"fresh"` → `get_peft_model(model, config)`.
  - `"resume"` → `PeftModel.from_pretrained(model, resume_path, is_trainable=True)`.
- `apply_kbit_preparation(model, gradient_checkpointing)` runs
  `prepare_model_for_kbit_training` for the QLoRA path. MUST be called
  BEFORE `get_peft_model` (audit risk noted in sprint spec).

The trainer's higher-level orchestrator composes these — we keep the
individual functions small + testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec


def build_lora_config(
    spec: BaseModelSpec,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    tokenizer_grew: bool,
) -> Any:
    """Return a `peft.LoraConfig` sized for `spec`.

    If `tokenizer_grew=True` (Sprint 07 bringup added a new pad token),
    we MUST train the embedding + lm_head alongside the LoRA deltas —
    otherwise the new embedding row is undefined. `modules_to_save`
    inflates the adapter checkpoint size substantially; surfacing this
    at the LoRA level keeps the tradeoff auditable (CLAUDE.md pitfall
    #4 / audit F02).
    """
    from peft import LoraConfig, TaskType

    modules_to_save = ["embed_tokens", "lm_head"] if tokenizer_grew else None

    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(spec.target_modules),
        modules_to_save=modules_to_save,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def apply_kbit_preparation(model: Any, *, gradient_checkpointing: bool) -> Any:
    """Run `prepare_model_for_kbit_training` for the QLoRA path.

    Required before `get_peft_model` on a 4-bit-loaded base model.
    Returns the prepared model (PEFT mutates in place but also
    returns it for chaining convenience).
    """
    from peft import prepare_model_for_kbit_training

    return prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )


def build_or_resume_adapter(
    base_model: Any,
    spec: BaseModelSpec,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    tokenizer_grew: bool,
    mode: Literal["fresh", "resume"],
    resume_path: Any = None,
    use_qlora: bool = False,
    gradient_checkpointing: bool = False,
) -> Any:
    """Wrap `base_model` in a PEFT adapter, fresh or resumed.

    On QLoRA, `prepare_model_for_kbit_training` is applied before
    either path — this is the order PEFT + bnb both require.
    """
    from peft import PeftModel, get_peft_model

    if use_qlora:
        base_model = apply_kbit_preparation(
            base_model, gradient_checkpointing=gradient_checkpointing
        )

    if mode == "resume":
        if resume_path is None:
            raise ValueError("resume mode requires `resume_path`")
        return PeftModel.from_pretrained(base_model, str(resume_path), is_trainable=True)

    config = build_lora_config(
        spec,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        tokenizer_grew=tokenizer_grew,
    )
    return get_peft_model(base_model, config)
