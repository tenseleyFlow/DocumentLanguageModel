"""Audio-language inference loader for `dlm prompt --audio`.

Parallel to `vl_loader.py`. Loads an audio-LM base + adapter + the
full `AutoProcessor` (ProcessorMixin bundling tokenizer + feature
extractor). QLoRA plumbing deferred to a follow-up — the Qwen2-Audio
fp16 checkpoint is ~15 GB which doesn't fit on Apple Silicon
consumer memory without 4-bit anyway, but wiring bitsandbytes through
the audio path safely needs its own slow-test coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.inference.loader import resolve_adapter_path
from dlm.inference.plan import InferencePlan
from dlm.train.loader import _AUDIO_MODEL_CLASSES

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.store.paths import StorePath


@dataclass(frozen=True)
class LoadedAudioInference:
    """Result of `load_for_audio_inference`."""

    model: Any
    processor: Any
    plan: InferencePlan
    adapter_path: Path


def load_for_audio_inference(  # pragma: no cover
    store: StorePath,
    spec: BaseModelSpec,
    caps: Any,
    *,
    adapter_name: str | None = None,
) -> LoadedAudioInference:
    """Resolve plan + load audio-LM base + adapter + processor.

    Pragma'd from unit coverage — exercises class-named model load +
    `AutoProcessor.from_pretrained` over real HF weights. Covered by
    the Sprint 35.2 slow integration test (T12).
    """
    if spec.modality != "audio-language":
        raise ValueError(
            f"load_for_audio_inference: {spec.key!r} is modality={spec.modality!r}; "
            "use load_for_inference for text bases or load_for_vl_inference for VL"
        )

    if spec.architecture not in _AUDIO_MODEL_CLASSES:
        raise ValueError(
            f"load_for_audio_inference: no audio-LM loader wired for architecture "
            f"{spec.architecture!r}; add a mapping to _AUDIO_MODEL_CLASSES"
        )

    adapter_path = resolve_adapter_path(store, adapter_name=adapter_name)

    import transformers
    from transformers import AutoProcessor

    from dlm.inference.plan import resolve_inference

    plan = resolve_inference(adapter_path, caps)
    dtype = _torch_dtype_for(plan.precision)

    model_cls = getattr(transformers, _AUDIO_MODEL_CLASSES[spec.architecture])
    base = model_cls.from_pretrained(
        spec.hf_id,
        revision=spec.revision,
        torch_dtype=dtype,
        attn_implementation=plan.attn_implementation,
    )

    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    # Processor is pinned on the base revision — same rationale as VL.
    processor = AutoProcessor.from_pretrained(spec.hf_id, revision=spec.revision)

    return LoadedAudioInference(
        model=model,
        processor=processor,
        plan=plan,
        adapter_path=adapter_path,
    )


def _torch_dtype_for(precision: str) -> Any:  # pragma: no cover
    try:
        import torch
    except ImportError:
        return precision
    lookup = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    return lookup.get(precision, torch.float16)
