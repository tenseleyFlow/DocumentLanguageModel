"""Vision-language generation path for `dlm prompt --image`.

Mirrors `dlm.inference.generate` but drives an HF `AutoProcessor`
(not a bare tokenizer) + `AutoModelForImageTextToText` through a
prompt that carries one or more image placeholders.

Shape contract matches what TRL 1.2's
`DataCollatorForVisionLanguageModeling` emits at training time: the
user's text carries the base's `image_token` placeholder (e.g.
`<image>`) and the processor expands each occurrence into the base's
`num_image_tokens` slots. This keeps prompt-time input aligned with
training-time input — the same lesson the text path learned with
`format_chat_prompt`.

Heavy imports (`PIL`, `torch`) defer inside the functions so importing
this module stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlm.inference.generate import DEFAULT_MAX_NEW_TOKENS, build_generate_kwargs


def format_vl_prompt(
    prompt: str,
    *,
    image_token: str,
    num_images: int,
) -> str:
    """Build the VL-aware prompt text.

    When the user's prompt already contains `image_token`, pass it
    through — they explicitly placed the image. Otherwise prepend one
    `image_token` per image so the processor can slot the pixels in
    before the text; trailing newline separates the image block from
    the user's question the way every VL chat template does.

    This matches `sections_to_rows`' IMAGE emission at training time:
    `"<image>\\n<caption>"` — training and prompt-time input see the
    same token order.
    """
    if image_token in prompt:
        return prompt
    tokens = image_token * num_images
    return f"{tokens}\n{prompt}" if prompt else tokens


def load_images(paths: list[Path]) -> list[Any]:
    """Open each path as a PIL.Image in RGB mode.

    Raises `FileNotFoundError` on missing paths; a PIL `UnidentifiedImageError`
    on files that aren't a decodable image. Both bubble up to the CLI
    which converts them into typer exits.
    """
    from PIL import Image

    images: list[Any] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        with Image.open(path) as pil:
            pil.load()
            images.append(pil.convert("RGB"))
    return images


def generate_vl(  # pragma: no cover
    model: Any,
    processor: Any,
    prompt: str,
    images: list[Any],
    *,
    image_token: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
) -> str:
    """Render VL prompt, run generation, decode response-only tokens.

    `processor` is an `AutoProcessor` for a VL base. `images` is a
    list of PIL.Image objects, one per `image_token` occurrence in
    `prompt` (or pre-prepended for the user). `image_token` comes from
    the base's `VlPreprocessorPlan`.

    Pragma'd from unit coverage because it calls `model.generate` on a
    real HF VL model; covered by the slow-marked integration test.
    """
    import torch

    formatted = format_vl_prompt(prompt, image_token=image_token, num_images=len(images))
    inputs = processor(
        images=images,
        text=formatted,
        return_tensors="pt",
    ).to(model.device)
    input_len = int(inputs["input_ids"].shape[-1])

    gen_kwargs = build_generate_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    tokenizer = getattr(processor, "tokenizer", processor)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **gen_kwargs,
            pad_token_id=tokenizer.pad_token_id,
        )

    response_tokens = output[0, input_len:]
    decoded = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return str(decoded)
