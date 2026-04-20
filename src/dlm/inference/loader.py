"""Adapter loader for inference (`dlm prompt`).

Given a `StorePath` and the current host's `Capabilities`, resolve an
`InferencePlan` and load the PEFT model + tokenizer ready for
`generate()`. Two paths:

- **4-bit QLoRA path** (CUDA + bnb installed + adapter was QLoRA-trained):
  `AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb)`
  then `PeftModel.from_pretrained(base, adapter_dir)`.
- **fp16 / bf16 path** (everything else, including the F05 "CUDA-saved
  QLoRA resumed on Apple Silicon" case): `AutoModelForCausalLM` at the
  plan's `precision`, then adapter load. Dequantization for a 4-bit-
  trained adapter loaded without bnb happens implicitly: the saved
  LoRA delta weights are already in fp16; loading the BASE at fp16
  (not 4-bit) is the correct behavior. The adapter adds a small
  fp16 residual on top of a fp16 base.

The tokenizer is loaded from the **adapter directory**, not the
`store.cache/`, because Sprint 07's bringup persists the final
tokenizer state (including `<|pad|>` additions) into the adapter dir
at training-end. This is the cross-sprint contract F02 depends on.

Heavy imports are deferred; the orchestration logic that picks args,
paths, and dtypes is unit-testable without HF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.inference.errors import AdapterNotFoundError
from dlm.inference.plan import InferencePlan

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.store.paths import StorePath


@dataclass(frozen=True)
class LoadedInference:
    """Result of `load_for_inference`."""

    model: Any  # PeftModel — Any to avoid pulling peft into type stubs
    tokenizer: Any
    plan: InferencePlan
    adapter_path: Path


def build_load_kwargs(
    spec: BaseModelSpec,
    plan: InferencePlan,
    *,
    has_bitsandbytes: bool,
) -> dict[str, Any]:
    """Assemble `AutoModelForCausalLM.from_pretrained` kwargs for `plan`.

    Extracted so unit tests can verify the config-assembly logic
    without actually loading a model. The real loader calls this plus
    the HF API; this function returns the dict, nothing more.

    - QLoRA path: `quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...)`.
    - Dequantize path: plain `torch_dtype=...`; no quantization config.
    - Plain LoRA / fp: `torch_dtype=...`.
    """
    kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "attn_implementation": plan.attn_implementation,
    }

    if not plan.dequantize_on_load and has_bitsandbytes and plan.precision in ("bf16", "fp16"):
        # Only reach here on the real 4-bit CUDA+bnb path.
        from transformers import BitsAndBytesConfig  # pragma: no cover

        compute_dtype = _torch_dtype_for(plan.precision)  # pragma: no cover
        kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]  # pragma: no cover
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = _torch_dtype_for(plan.precision)

    return kwargs


def _torch_dtype_for(precision: str) -> Any:
    """Map precision string to `torch.dtype`.

    Isolated so unit tests can call `build_load_kwargs` with a string
    result (they assert the key shape, not the exact dtype object) while
    the real path still gets a torch.dtype.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return precision

    lookup = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    return lookup.get(precision, torch.float16)


def resolve_adapter_path(
    store: StorePath, *, adapter_name: str | None
) -> Path:
    """Return the on-disk adapter version dir for inference.

    Single entry point for both the flat (unnamed) and named-adapter
    layouts. Raises `AdapterNotFoundError` with a path-appropriate
    hint when `current.txt` is missing or empty — the most common
    "haven't trained yet" failure mode.
    """
    if adapter_name is None:
        adapter_path = store.resolve_current_adapter()
        pointer = store.adapter_current_pointer
    else:
        adapter_path = store.resolve_current_adapter_for(adapter_name)
        pointer = store.adapter_current_pointer_for(adapter_name)
    if adapter_path is None or not adapter_path.exists():
        hint = (
            f"no adapter under {pointer}; "
            f"has `dlm train` run successfully"
            f"{f' for adapter {adapter_name!r}' if adapter_name else ''}?"
        )
        raise AdapterNotFoundError(hint)
    return adapter_path


def load_for_inference(  # pragma: no cover
    store: StorePath,
    spec: BaseModelSpec,
    caps: Any,
    *,
    adapter_name: str | None = None,
) -> LoadedInference:
    """Resolve plan + load base + adapter + tokenizer.

    Pragma'd from unit coverage because it calls `AutoModelForCausalLM.from_pretrained`
    and `PeftModel.from_pretrained`, which each need ~5 seconds and a
    real HF cache. Covered by Sprint 10's slow-marked integration test.

    `adapter_name`, when provided, targets the named multi-adapter
    layout (`adapter/<name>/current.txt`). When `None`, uses the flat
    single-adapter layout.
    """
    adapter_path = resolve_adapter_path(store, adapter_name=adapter_name)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dlm.inference.plan import resolve_inference

    plan = resolve_inference(adapter_path, caps)
    has_bnb = bool(getattr(caps, "has_bitsandbytes", False))
    kwargs = build_load_kwargs(spec, plan, has_bitsandbytes=has_bnb)

    base = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)

    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    # Tokenizer from the adapter dir — source of truth after any
    # vocab growth (Sprint 07 bringup contract).
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

    return LoadedInference(
        model=model,
        tokenizer=tokenizer,
        plan=plan,
        adapter_path=adapter_path,
    )
