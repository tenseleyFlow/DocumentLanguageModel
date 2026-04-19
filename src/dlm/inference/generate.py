"""`generate(model, tokenizer, prompt, **kwargs) -> str`.

Thin wrapper around HF `model.generate` with deterministic defaults.

Deterministic generation requires ALL of:
- `do_sample=False`
- `num_beams=1`
- `temperature=0.0` (technically moot when do_sample=False, but
  some HF code paths still read it — belt and braces)
- The model's cuDNN flags set to deterministic mode (Sprint 09
  `determinism.seed_everything` handles this at `dlm train` time)

When the caller passes `temperature > 0`, we flip `do_sample=True`
automatically — otherwise a non-zero temperature is silently ignored
by HF on `do_sample=False` and users get confused.

`generate()` returns the model's response only, not the prompt echo —
the HF API prepends the prompt tokens to the output sequence; we strip
them here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

DEFAULT_MAX_NEW_TOKENS = 256


def build_generate_kwargs(
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
) -> dict[str, Any]:
    """Assemble the kwargs dict for `model.generate`.

    Extracted so the argument-resolution logic is unit-testable
    without instantiating a model. Rules:
    - `temperature == 0` → `do_sample=False`, `num_beams=1` (deterministic).
    - `temperature > 0` → `do_sample=True`, keep temperature.
    - `top_p`/`top_k`/`repetition_penalty` only included when set, so
      the HF defaults pass through.
    """
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0.0, got {temperature}")

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
    }

    if temperature == 0.0:
        kwargs["do_sample"] = False
        kwargs["num_beams"] = 1
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k

    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty

    return kwargs


def format_chat_prompt(tokenizer: Any, prompt: str) -> str:  # pragma: no cover
    """Render `prompt` through the tokenizer's chat template.

    Matches how `dlm.data.formatter.make_formatting_func` shapes rows
    during training, so prompt-time input sees the same template as
    training-time input. If the tokenizer has no chat_template, fall
    back to the raw string — this is surfaced by the caller's UX; we
    don't want `dlm prompt` to refuse just because a base lacks a
    chat_template (the user may have supplied one explicitly elsewhere).
    """
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(rendered, str):
            return rendered
    return prompt


def generate(  # pragma: no cover
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
) -> str:
    """Render `prompt`, run generation, decode response-only tokens.

    Pragma'd from unit coverage because it calls `model.generate`.
    Covered by Sprint 10's slow-marked integration test.
    """
    import torch

    formatted = format_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = int(inputs["input_ids"].shape[-1])

    gen_kwargs = build_generate_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **gen_kwargs,
            pad_token_id=tokenizer.pad_token_id,
        )

    response_tokens = output[0, input_len:]
    decoded = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return str(decoded)
