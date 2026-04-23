"""Vision-language runtime contract checks.

The registry can legitimately know about a VL family before DLM owns a
complete runtime path for it. This module centralizes the fail-fast
checks for those gaps so train / prompt / export snapshot all surface
the same explanation instead of failing later inside transformers or a
custom model forward pass.
"""

from __future__ import annotations

from typing import Any

from dlm.modality.errors import ProcessorContractError


def ensure_supported_vl_runtime(spec: Any) -> None:
    """Refuse VL families whose runtime contract DLM has not wired yet."""
    if getattr(spec, "modality", None) != "vision-language":
        return
    if getattr(spec, "architecture", None) != "InternVLChatModel":
        return
    raise ProcessorContractError(
        f"base {spec.key!r} is an InternVL-family VL model. On the current "
        "transformers stack, `AutoProcessor.from_pretrained(...)` resolves to a "
        "tokenizer-only object, while the upstream runtime also expects dynamic "
        "`<image>` → `<img><IMG_CONTEXT>*...` expansion and `image_flags` on the "
        "forward path. DLM has not wired that custom processor/collator contract "
        "yet, so prompt/train/HF-snapshot export refuse this family instead of "
        "pretending the generic VL path is enough."
    )


def validate_loaded_vl_processor(spec: Any, processor: Any) -> Any:
    """Return `processor` or raise when it can't drive VL preprocessing."""
    if getattr(spec, "modality", None) != "vision-language":
        return processor

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        return processor

    if getattr(spec, "architecture", None) == "InternVLChatModel":
        ensure_supported_vl_runtime(spec)

    raise ProcessorContractError(
        f"base {spec.key!r} loaded a vision-language processor without an "
        "`image_processor` attribute. DLM's VL runtime expects a processor that "
        "owns both tokenization and image preprocessing; refusing before the "
        "generic VL collator silently drops the image path."
    )
