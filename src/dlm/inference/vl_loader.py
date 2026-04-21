"""VL inference loader (base + processor + adapter) for `dlm prompt --image`.

Parallel to `dlm.inference.loader` — the text path loads
`AutoModelForCausalLM` + a tokenizer; this path loads
`AutoModelForImageTextToText` + the full `AutoProcessor` (ProcessorMixin).
QLoRA is not plumbed through the VL path in v1: PaliGemma fp16 fits
on 16 GB MPS, and the bitsandbytes + VL weight loading combination
isn't exercised anywhere in our test matrix yet — Sprint 35.3 or a
dedicated audit can thread it when the need surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.inference.loader import resolve_adapter_path
from dlm.inference.plan import InferencePlan

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.store.paths import StorePath


@dataclass(frozen=True)
class LoadedVlInference:
    """Result of `load_for_vl_inference`."""

    model: Any
    processor: Any
    plan: InferencePlan
    adapter_path: Path


def load_for_vl_inference(  # pragma: no cover
    store: StorePath,
    spec: BaseModelSpec,
    caps: Any,
    *,
    adapter_name: str | None = None,
) -> LoadedVlInference:
    """Resolve plan + load VL base + adapter + processor.

    Pragma'd from unit coverage: exercises `AutoModelForImageTextToText.from_pretrained`
    and `AutoProcessor.from_pretrained` over real HF weights. Covered
    by the Sprint 35 v1 slow integration test (T12).
    """
    if spec.modality != "vision-language":
        raise ValueError(
            f"load_for_vl_inference: {spec.key!r} is modality={spec.modality!r}; "
            "use load_for_inference for text bases"
        )

    adapter_path = resolve_adapter_path(store, adapter_name=adapter_name)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    from dlm.inference.plan import resolve_inference

    plan = resolve_inference(adapter_path, caps)
    dtype = _torch_dtype_for(plan.precision)

    base = AutoModelForImageTextToText.from_pretrained(
        spec.hf_id,
        revision=spec.revision,
        torch_dtype=dtype,
        attn_implementation=plan.attn_implementation,
    )

    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    # Processor comes from the pinned base (not the adapter dir) because
    # VL adapters don't snapshot the processor — pixel-path config is
    # deterministic per base revision.
    processor = AutoProcessor.from_pretrained(spec.hf_id, revision=spec.revision)

    return LoadedVlInference(
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
