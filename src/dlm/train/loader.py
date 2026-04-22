"""Base-model loader with optional QLoRA quantization.

`load_base_model(spec, plan)` loads the HF model with the dtype +
quantization dictated by the `TrainingPlan`:

- `use_qlora=True` → `BitsAndBytesConfig(load_in_4bit=True, nf4,
  compute_dtype=plan.compute_dtype, double_quant=True)`. Only valid
  on CUDA; the hardware doctor enforces this upstream.
- `use_qlora=False` → plain load at `plan.load_dtype` (bf16/fp16 on
  GPU, fp32 on CPU).

All heavy imports are deferred inside the function. `bitsandbytes` is
only imported on the QLoRA branch — this lets the module import
cleanly on Apple Silicon (where bnb isn't installable) so long as the
caller doesn't ask for QLoRA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.hardware.plan import TrainingPlan


def load_base_model(spec: BaseModelSpec, plan: TrainingPlan) -> Any:  # pragma: no cover
    """Return an HF `PreTrainedModel` loaded per `plan`.

    Text bases load via `AutoModelForCausalLM`; vision-language bases
    load via `AutoModelForImageTextToText`. Audio-language bases lack
    a generic `AutoModelFor*TextToText` class in transformers 5.x, so
    they load via the architecture-class named on the spec
    (`spec.architecture`) — e.g. `Qwen2AudioForConditionalGeneration`.

    All three share the quantization + dtype + attention wiring — the
    only delta is which class is imported + instantiated.

    Covered by the slow-marked integration tests; instantiating even a
    tiny HF model is >2 s so this stays out of the unit suite.
    """
    dtype = _resolve_torch_dtype(plan.precision)

    kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "dtype": dtype,
        "attn_implementation": plan.attn_implementation,
    }

    # Spec-level opt-in for bases whose HF class lives in the model's
    # own repo (e.g. InternVL2's `InternVLChatModel`). Only flows
    # through when explicitly declared `True` on the registry entry —
    # never inferred, never user-toggled at runtime.
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True

    if plan.use_qlora:
        kwargs["quantization_config"] = _build_bnb_config(plan)

    from dlm.modality import modality_for

    dispatch = modality_for(spec)
    if dispatch.accepts_images:
        # Bases with `trust_remote_code=True` often aren't registered
        # with AutoModelForImageTextToText (that's the whole reason —
        # their class lives in the repo, not transformers). Fall back
        # to `AutoModel.from_pretrained(trust_remote_code=True)` which
        # reads the repo's config + dispatches to the custom class.
        from transformers import AutoModel, AutoModelForImageTextToText

        if spec.trust_remote_code:
            return AutoModel.from_pretrained(spec.hf_id, **kwargs)
        return AutoModelForImageTextToText.from_pretrained(spec.hf_id, **kwargs)

    if dispatch.accepts_audio:
        # No AutoModelForAudioTextToText in transformers 5.x; resolve
        # the class name from `spec.architecture` so adding a new audio
        # base is a registry edit, not a loader patch.
        model_cls = _resolve_audio_model_class(spec.architecture)
        return model_cls.from_pretrained(spec.hf_id, **kwargs)

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)


def load_processor(spec: BaseModelSpec) -> Any:  # pragma: no cover
    """Return the HF `ProcessorMixin` for a media-modality base.

    Both VL and audio-language bases carry an `AutoProcessor` that
    bundles the tokenizer + modality-specific feature extractor +
    chat template. Text-modality specs raise — callers should branch
    on `spec.modality` before reaching this path.

    Threads `trust_remote_code=True` through when the spec opts in
    (e.g. InternVL2's processor lives in the model repo too).
    """
    if spec.modality not in ("vision-language", "audio-language"):
        raise ValueError(
            f"load_processor: {spec.key!r} is modality='{spec.modality}'; "
            "processors are only loaded for media bases (vision-language / audio-language)"
        )
    from dlm.base_models._typed_shims import load_auto_processor

    kwargs: dict[str, Any] = {"revision": spec.revision}
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True
    return load_auto_processor(spec.hf_id, **kwargs)


_AUDIO_MODEL_CLASSES: dict[str, str] = {
    # Maps `BaseModelSpec.architecture` → transformers class name.
    # Sprint 35.2 v1 ships Qwen2-Audio only; add new entries here when
    # more audio-LM families land in the registry.
    "Qwen2AudioForConditionalGeneration": "Qwen2AudioForConditionalGeneration",
}


def _resolve_audio_model_class(architecture: str) -> Any:  # pragma: no cover
    """Import the architecture-class named on the spec.

    Deferred import so the module stays cheap for text-only callers.
    Unknown archs raise — the registry drift check should catch this
    before the trainer does, but we surface a readable error in case
    a new entry landed without a loader update.
    """
    if architecture not in _AUDIO_MODEL_CLASSES:
        raise ValueError(
            f"load_base_model: no audio-LM loader wired for architecture "
            f"{architecture!r}; add a mapping to _AUDIO_MODEL_CLASSES"
        )
    import transformers

    cls_name = _AUDIO_MODEL_CLASSES[architecture]
    return getattr(transformers, cls_name)


def _build_bnb_config(plan: TrainingPlan) -> Any:  # pragma: no cover
    """Canonical NF4 double-quant 4-bit config.

    Deferred import keeps bitsandbytes off the happy path on non-CUDA
    dev machines. The doctor refuses QLoRA without bnb installed
    (CLAUDE.md pitfall #6 context).
    """
    from transformers import BitsAndBytesConfig

    # QLoRA always pairs 4-bit storage with a higher-precision compute
    # dtype; the plan owns the pick (bf16 on Ampere+, fp16 otherwise).
    compute_dtype_name = plan.quant_compute_dtype or plan.precision
    compute_dtype = _resolve_torch_dtype(compute_dtype_name)

    return BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def _resolve_torch_dtype(name: str | Any) -> Any:  # pragma: no cover
    """Map `"bf16" | "fp16" | "fp32"` (or a torch.dtype) to a torch.dtype.

    Only called from `load_base_model` / `_build_bnb_config` (both of
    which are also pragma'd — covered by slow-marked integration tests).
    """
    import torch

    if not isinstance(name, str):
        return name
    lookup = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in lookup:
        raise ValueError(f"unknown dtype {name!r}; expected bf16/fp16/fp32")
    return lookup[name]
