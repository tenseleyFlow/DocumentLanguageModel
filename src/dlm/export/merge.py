"""Merged-GGUF export path — where LoRA deltas fuse into the base.

CLAUDE.md pitfall #3: `merge_and_unload` on a 4-bit QLoRA base is
precision-unsafe. The canonical safety decision now lives in
`dlm.export.precision_safety`; this module only hosts the heavy HF
merge work plus a tiny pure helper used by property tests.

The actual HF work (loading the base in fp16, calling `merge_and_unload`,
saving to a tmpdir) is pragma'd from unit coverage: it needs a real
model and takes minutes on CI. `check_merge_safety` — the pure
decision function — is fully covered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlm.export.plan import ExportPlan


def check_merge_safety(plan: ExportPlan, *, was_qlora: bool) -> None:
    """Pure-python truth-table helper — no subprocess, no HF.

    Main export entry points use `dlm.export.precision_safety` so the
    adapter-metadata probe and the merged-export gate stay together.
    This wrapper remains for focused unit/property tests over the
    boolean truth table itself.
    """
    plan.assert_merge_safe(was_qlora=was_qlora)


def perform_merge(  # pragma: no cover
    adapter_dir: Path,
    out_hf_dir: Path,
    *,
    was_qlora: bool,
    cached_base_dir: Path,
) -> None:
    """Load base + adapter, merge_and_unload, save merged HF dir.

    Pragma'd from unit coverage: instantiating a real HF model is
    >5s per test and requires a cached checkpoint. Exercised by the
    slow-marked integration test.

    Never entered without `check_merge_safety()` having passed first —
    the runner enforces the order. `was_qlora=True` additionally
    requires the plan's `--dequantize` flag to have been confirmed.

    `cached_base_dir` is the HF snapshot dir produced by
    `base_models.downloader.download_spec(spec).path`; we pass it in
    (rather than re-`from_pretrained(spec.hf_id, ...)`) so the merge
    path reuses the already-verified, sha256-pinned cache and never
    touches the network at export time.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Both QLoRA and plain-LoRA adapters merge onto the upstream fp16
    # base weights. For QLoRA, loading in fp16 (rather than re-running
    # bnb 4-bit quantization) is the dequantization — the base weights
    # in the cache are already fp16 upstream and LoRA deltas merge at
    # native precision. `was_qlora` is kept in the signature for
    # downstream logging / audit trails.
    _ = was_qlora
    torch_dtype = torch.float16

    base = AutoModelForCausalLM.from_pretrained(
        str(cached_base_dir),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    peft: Any = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft.merge_and_unload()

    out_hf_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_hf_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), local_files_only=True)
    tokenizer.save_pretrained(str(out_hf_dir))


def perform_vl_merge(  # pragma: no cover
    adapter_dir: Path,
    out_hf_dir: Path,
    *,
    cached_base_dir: Path,
) -> None:
    """VL-aware merge: `AutoModelForImageTextToText` + full processor save.

    Parallel to `perform_merge` but uses the image-text-to-text class so
    the vision tower travels with the merged output (upstream
    `convert_hf_to_gguf.py` drops ViT tensors for Qwen2-VL at our
    pinned tag — the ViT runs through Ollama's preprocessor path — but
    `from_pretrained` still needs the VL class to reconstruct the full
    graph before `merge_and_unload`).

    LoRA adapters for VL bases should target language-model projections
    only (enforced by `preflight.check_vl_target_modules_lm_only`), so
    `merge_and_unload()` touches LM weights exclusively; vision-tower
    weights are saved unmodified.

    `processor.save_pretrained` (not just tokenizer) writes the
    tokenizer + image_processor + processor config together — every
    piece a recipient needs to re-hydrate.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base = AutoModelForImageTextToText.from_pretrained(
        str(cached_base_dir),
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    peft: Any = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft.merge_and_unload()

    out_hf_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_hf_dir))

    processor = AutoProcessor.from_pretrained(str(cached_base_dir), local_files_only=True)  # type: ignore[no-untyped-call]
    processor.save_pretrained(str(out_hf_dir))
