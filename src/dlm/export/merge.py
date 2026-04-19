"""Merged-GGUF export path — where LoRA deltas fuse into the base.

CLAUDE.md pitfall #3: `merge_and_unload` on a 4-bit QLoRA base is
precision-unsafe. This module is the single enforcement point for
the rule — `ExportPlan.assert_merge_safe(was_qlora=...)` is called
here, and the fp16-merge path is only entered when the user has
explicitly confirmed via `--dequantize`.

The actual HF work (loading the base in fp16, calling `merge_and_unload`,
saving to a tmpdir) is pragma'd from unit coverage: it needs a real
model and takes minutes on CI. `check_merge_safety` — the pure
decision function — is fully covered.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.export.plan import ExportPlan

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec


def check_merge_safety(plan: ExportPlan, *, was_qlora: bool) -> None:
    """Pure-python safety gate — no subprocess, no HF, just rule enforcement.

    Delegates to `ExportPlan.assert_merge_safe` so there's exactly one
    rule in the codebase. This wrapper exists so callers read
    "merge.check_merge_safety" at the call site, which reads like a
    safety check rather than a plan mutation.
    """
    plan.assert_merge_safe(was_qlora=was_qlora)


def perform_merge(  # pragma: no cover
    spec: BaseModelSpec,
    adapter_dir: Path,
    out_hf_dir: Path,
    *,
    was_qlora: bool,
) -> None:
    """Load base + adapter, merge_and_unload, save merged HF dir.

    Pragma'd from unit coverage: instantiating a real HF model is
    >5s per test and requires a cached checkpoint. Exercised by the
    slow-marked integration test.

    Never entered without `check_merge_safety()` having passed first —
    the runner enforces the order.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Dequantize path: base was trained 4-bit, but we load fp16 here
    # so the merge math happens at native precision. Acceptable only
    # because the plan's `--dequantize` flag was checked.
    torch_dtype = torch.float16 if was_qlora else torch.float16

    base = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        revision=spec.revision,
        torch_dtype=torch_dtype,
    )
    peft: Any = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft.merge_and_unload()

    out_hf_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_hf_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(out_hf_dir))
