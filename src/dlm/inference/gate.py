"""Inference-time gate hook — compute prompt embedding, run gate, return
per-adapter weights.

The gate is a small MLP trained post-SFT (see ``dlm.train.gate``). At
inference time the caller:

1. Tokenizes the prompt.
2. Runs the base model with all adapters disabled, mean-pools the
   last hidden state → a prompt embedding.
3. Feeds the embedding through the loaded gate → a dict of
   ``{adapter_name: weight}`` suitable for PEFT's
   ``set_adapter_weights``.

The gate forward is ~1ms on MPS for the default shape — negligible
vs generation cost, and the embedding is computed once per request
(not per token).

Uniform-mode stores (cold-start fallback) skip steps 1-3 entirely:
``weights_for_prompt`` returns ``1/N`` for each declared adapter
without touching the base model at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dlm.train.gate import Gate, GateMetadata, load_gate

if TYPE_CHECKING:
    import torch

    from dlm.store.paths import StorePath


@dataclass(frozen=True)
class GateHandle:
    """Loaded gate + metadata, ready to route prompts.

    ``gate`` is ``None`` when the store recorded a uniform-mode
    fallback — callers use ``weights_for_prompt`` which dispatches
    internally.
    """

    gate: Gate | None
    metadata: GateMetadata

    @property
    def adapter_names(self) -> tuple[str, ...]:
        return self.metadata.adapter_names

    @property
    def is_uniform(self) -> bool:
        return self.metadata.mode == "uniform"


def load_gate_handle(store: StorePath) -> GateHandle:
    """Load the gate for inference. Raises ``GateConfigError`` if the
    store's gate config is missing or malformed; callers that want a
    "no gate = uniform default" policy should catch and fall back."""
    gate, meta = load_gate(store)
    return GateHandle(gate=gate, metadata=meta)


def embed_prompt(
    *,
    prompt: str,
    tokenizer: object,
    base_model: object,
    max_length: int = 512,
) -> torch.Tensor:
    """Tokenize + forward + mean-pool last hidden state.

    Returns a 1-D tensor of shape ``(hidden_dim,)`` on CPU. The base
    model must be called with adapters disabled (the caller is
    responsible — typically via ``peft_model.disable_adapter()``).

    ``tokenizer`` / ``base_model`` are HuggingFace-style objects; we
    avoid importing their concrete types to keep this module light.
    """
    import torch

    assert callable(tokenizer), "tokenizer must be callable"
    assert hasattr(base_model, "forward"), "base_model must have forward()"

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    # Match the model's device so we don't pay a host↔device hop per
    # generation call. Parameters may be empty (CPU-only tiny fixtures).
    try:
        device = next(base_model.parameters()).device  # type: ignore[attr-defined]
    except (StopIteration, AttributeError):
        device = torch.device("cpu")
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = base_model(  # type: ignore[operator]
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = outputs.hidden_states[-1]  # (1, seq, hidden)
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        pooled = hidden.mean(dim=1)
    result: torch.Tensor = pooled.squeeze(0).to(torch.float32).cpu()
    return result


def weights_for_prompt(
    handle: GateHandle,
    *,
    prompt: str,
    tokenizer: object,
    base_model: object,
) -> dict[str, float]:
    """Return ``{adapter_name: weight}`` for this prompt.

    Uniform-mode gates short-circuit to ``1/N`` without running the
    base-model forward pass; trained gates embed the prompt and run
    the learned MLP. Weights always sum to 1.0 (softmax output or
    explicit uniform).
    """
    import torch

    n = len(handle.adapter_names)
    if handle.is_uniform or handle.gate is None:
        return dict.fromkeys(handle.adapter_names, 1.0 / n)

    embedding = embed_prompt(prompt=prompt, tokenizer=tokenizer, base_model=base_model)
    if embedding.shape[0] != handle.metadata.input_dim:
        from dlm.train.gate.errors import GateConfigError

        raise GateConfigError(
            f"prompt embedding dim {embedding.shape[0]} != gate input_dim "
            f"{handle.metadata.input_dim} (base model mismatch?)"
        )
    with torch.no_grad():
        probs = handle.gate(embedding.unsqueeze(0)).squeeze(0)
    return {name: float(probs[i].item()) for i, name in enumerate(handle.adapter_names)}


def uniform_weights(adapter_names: tuple[str, ...]) -> dict[str, float]:
    """Public helper for `--gate off` — no gate consulted."""
    n = len(adapter_names)
    if n == 0:
        return {}
    return dict.fromkeys(adapter_names, 1.0 / n)
