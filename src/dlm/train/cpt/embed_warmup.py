"""Embedding-layer warm-up for continued pretraining.

For domain-adaptive pretraining on prose whose vocabulary drifts from
the base's training distribution (code, domain jargon, archaic
English), it's sometimes useful to unfreeze `embed_tokens` + `lm_head`
for the first few hundred optimizer steps so the embeddings can absorb
the new token distribution, then refreeze and resume standard LoRA
training.

Two surfaces:

- `unfreeze_embeddings_for(model)` — a context manager for ad-hoc use
  (tests, REPL inspection). Restores the original `requires_grad` on
  exit. Idempotent on nested entries.
- `EmbedWarmupCallback(n_steps)` — an HF `TrainerCallback` that flips
  `requires_grad` on step 0 and restores on step N. This is what the
  trainer wires when `training.cpt.embed_warmup_steps > 0`.

Both attach to `embed_tokens` / `lm_head` via the HF helpers
`get_input_embeddings` and `get_output_embeddings` rather than
traversing the module tree by name — that keeps the utility portable
across architectures whose naming differs (Qwen vs Llama vs SmolLM).

When `embed_warmup_steps > 0`, the caller MUST also extend
`modules_to_save` on the LoRA config so the adapter persists the
warmed-up embedding rows. `extend_modules_to_save_for_embed_warmup`
handles the union with the `tokenizer_grew` case.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any


def _get_embedding_weights(model: Any) -> list[Any]:
    """Collect the parameter handles for the model's input + output embeddings.

    Some architectures tie weights (input and output share storage); we
    still return both handles so the caller's bookkeeping stays
    symmetrical. De-duplication by `id()` avoids double-restoring a
    tied weight.
    """
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()

    weights: list[Any] = []
    seen: set[int] = set()
    for module in (embed, head):
        if module is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        if id(weight) in seen:
            continue
        seen.add(id(weight))
        weights.append(weight)
    return weights


@contextmanager
def unfreeze_embeddings_for(model: Any) -> Generator[list[Any], None, None]:
    """Unfreeze `embed_tokens` + `lm_head` for the body of the block.

    Yields the list of param handles that were flipped so the caller
    can, e.g., reset gradients on them. On exit the original
    `requires_grad` values are restored verbatim.
    """
    weights = _get_embedding_weights(model)
    originals: list[bool] = [bool(w.requires_grad) for w in weights]
    for w in weights:
        w.requires_grad = True
    try:
        yield weights
    finally:
        for w, orig in zip(weights, originals, strict=True):
            w.requires_grad = orig


def extend_modules_to_save_for_embed_warmup(
    existing: Sequence[str] | None,
    *,
    embed_warmup_steps: int,
) -> list[str] | None:
    """Union `existing` with `{embed_tokens, lm_head}` when warm-up is on.

    When `embed_warmup_steps == 0`, pass through (returning the input
    as a list — or None if it was None). When positive, add both
    embedding module names without duplicating entries already present
    (e.g., from `tokenizer_grew`).

    Ordering: preserve existing order, then append any missing
    embedding modules. `adapter_config.json` isn't order-sensitive,
    but stable ordering makes the JSON diff-friendly across runs.
    """
    if embed_warmup_steps <= 0:
        return list(existing) if existing is not None else None

    out = list(existing) if existing is not None else []
    for name in ("embed_tokens", "lm_head"):
        if name not in out:
            out.append(name)
    return out


class EmbedWarmupCallback:  # pragma: no cover - exercised by slow integration
    """HF `TrainerCallback`: unfreeze embeddings for the first N global steps.

    We don't subclass `transformers.TrainerCallback` at import time (it
    would pull transformers into the unit-test path); duck typing is
    sufficient since HF only looks up the on_* methods by name.

    Refreezes the embeddings once `state.global_step` reaches
    `n_steps`. If training ends before then, `on_train_end` restores
    the originals as a safety net.
    """

    def __init__(self, model: Any, n_steps: int) -> None:
        if n_steps < 0:
            raise ValueError(f"n_steps must be non-negative, got {n_steps}")
        self.model = model
        self.n_steps = n_steps
        self._weights: list[Any] = []
        self._originals: list[bool] = []
        self._active: bool = False

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        if self.n_steps <= 0:
            return
        self._weights = _get_embedding_weights(self.model)
        self._originals = [bool(w.requires_grad) for w in self._weights]
        for w in self._weights:
            w.requires_grad = True
        self._active = True

    def _restore(self) -> None:
        if not self._active:
            return
        for w, orig in zip(self._weights, self._originals, strict=True):
            w.requires_grad = orig
        self._active = False

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        if self._active and state.global_step >= self.n_steps:
            self._restore()

    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        self._restore()
