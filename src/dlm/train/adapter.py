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

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dlm.train.errors import ResumeIntegrityError

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec

_EMBEDDING_MODULES = ("embed_tokens", "lm_head")


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


def verify_resume_tokenizer_compat(adapter_dir: Path, *, tokenizer_grew: bool) -> None:
    """Assert the saved adapter's `modules_to_save` agrees with the current tokenizer.

    Audit-04 M5: on resume, we load a LoRA adapter whose `adapter_config.json`
    was written under a particular tokenizer state. If the current run's
    tokenizer bringup grew the vocab but the saved adapter doesn't train
    embeddings (or vice versa), the resumed training will silently corrupt
    the `<|pad|>` row or fail to update a re-resized embedding table.

    Raises `ResumeIntegrityError` with actionable text on mismatch. Missing
    or unreadable `adapter_config.json` is treated as a mismatch (the
    checkpoint is broken regardless).
    """
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise ResumeIntegrityError(
            f"adapter directory {adapter_dir} is missing adapter_config.json; re-train fresh."
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeIntegrityError(
            f"adapter_config.json at {adapter_dir} is unreadable: {exc}"
        ) from exc

    saved_modules = config.get("modules_to_save") or []
    saved_has_embeddings = any(m in saved_modules for m in _EMBEDDING_MODULES)

    if saved_has_embeddings != tokenizer_grew:
        if tokenizer_grew:
            raise ResumeIntegrityError(
                "tokenizer state diverged from adapter state: current bringup added "
                "a new pad token (vocab grew), but the saved adapter did NOT train "
                "embeddings. Re-train fresh with `--fresh`."
            )
        raise ResumeIntegrityError(
            "tokenizer state diverged from adapter state: saved adapter was trained "
            f"with modules_to_save={saved_modules!r}, but the current tokenizer did "
            "not require vocab growth. Re-train fresh with `--fresh`."
        )


def apply_kbit_preparation(model: Any, *, gradient_checkpointing: bool) -> Any:  # pragma: no cover
    """Run `prepare_model_for_kbit_training` for the QLoRA path.

    Required before `get_peft_model` on a 4-bit-loaded base model.
    Covered by slow-marked integration tests — unit tests exercise
    `build_lora_config` directly.
    """
    from peft import prepare_model_for_kbit_training

    return prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )


def build_or_resume_adapter(  # pragma: no cover
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
        verify_resume_tokenizer_compat(Path(resume_path), tokenizer_grew=tokenizer_grew)
        return PeftModel.from_pretrained(base_model, str(resume_path), is_trainable=True)

    config = build_lora_config(
        spec,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        tokenizer_grew=tokenizer_grew,
    )
    return get_peft_model(base_model, config)
