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

    Covered by the slow-marked integration test in Sprint 09 rather
    than unit tests: instantiating even a tiny HF model is >2 s.
    """
    from transformers import AutoModelForCausalLM

    dtype = _resolve_torch_dtype(plan.precision)

    kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "torch_dtype": dtype,
        "attn_implementation": plan.attn_implementation,
    }

    if plan.use_qlora:
        kwargs["quantization_config"] = _build_bnb_config(plan)

    return AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)


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
