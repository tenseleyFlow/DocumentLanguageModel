"""Speculative-decoding draft model pairs.

Ollama ≥ 0.5 accepts a `PARAMETER draft_model <tag>` directive that
wires a small model as the speculative-decoding drafter for a large
target. On accepted tokens, latency drops 2–3× (llama.cpp benchmarks).

The draft must share the target's tokenizer AND chat template — a
single wrong token id in the draft's proposals invalidates the
speculation step. Tokenizer-pre labels + template dialect are the
two fields we can check at registry-build time. This module ships a
curated map of `(target_key → DraftPair)` entries; an import-time
validator asserts each pair's compatibility against `BASE_MODELS`.

Drafts are referenced by their *Ollama community tag* (e.g.
`qwen2.5:0.5b`) rather than a dlm-trained adapter. Users install
the draft once via `ollama pull <tag>`; every subsequent
speculative export of a compatible target reuses that install.

Launch set:

- `qwen2.5-3b` + `qwen2.5-coder-1.5b` → `qwen2.5:0.5b`
- `llama-3.2-3b` → `llama3.2:1b`
- `smollm2-1.7b` → `smollm2:360m`

Qwen 1.5B / Llama 1B / SmolLM2 sub-1B aren't in the target set —
the draft overhead dominates latency savings below ~1.5B targets.
Phi-3.5-mini has no smaller sibling in the launch registry, so it
doesn't get an auto-draft.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from dlm.base_models.schema import BaseModelSpec


@dataclass(frozen=True)
class DraftPair:
    """One speculative-decoding pairing.

    `target_key` → registry key of the larger, trained base.
    `draft_registry_key` → registry key of the smaller sibling (used
        for the template / tokenizer compatibility check).
    `upstream_ollama_tag` → the Ollama community tag users pull
        (`ollama pull <tag>`). Referenced literally in the emitted
        Modelfile's `PARAMETER draft_model` directive.
    `notes` → human-readable provenance for diagnostics + docs.
    """

    target_key: str
    draft_registry_key: str
    upstream_ollama_tag: str
    notes: str


DRAFT_PAIRS: Final[tuple[DraftPair, ...]] = (
    DraftPair(
        target_key="qwen2.5-3b",
        draft_registry_key="qwen2.5-0.5b",
        upstream_ollama_tag="qwen2.5:0.5b",
        notes="Qwen 2.5 family; ChatML + qwen2 BPE.",
    ),
    DraftPair(
        target_key="qwen2.5-coder-1.5b",
        draft_registry_key="qwen2.5-0.5b",
        upstream_ollama_tag="qwen2.5:0.5b",
        notes=(
            "Qwen 2.5 Coder shares vocab with Qwen 2.5 Instruct; the "
            "0.5B Instruct draft is the smallest compatible sibling."
        ),
    ),
    DraftPair(
        target_key="llama-3.2-3b",
        draft_registry_key="llama-3.2-1b",
        upstream_ollama_tag="llama3.2:1b",
        notes="Llama 3.2 family; llama3 template + llama-bpe.",
    ),
    DraftPair(
        target_key="smollm2-1.7b",
        draft_registry_key="smollm2-360m",
        upstream_ollama_tag="smollm2:360m",
        notes="SmolLM2 family; ChatML + smollm tokenizer.",
    ),
)


def resolve_draft(
    target_spec: BaseModelSpec,
    *,
    override: str | None = None,
    disabled: bool = False,
) -> str | None:
    """Return the Ollama tag to use as the draft model, or `None`.

    Precedence:
    - `disabled=True` → always `None` (user passed `--no-draft`).
    - `override` set → return it verbatim (user passed `--draft tag`).
    - Registry match on `target_spec.key` → return the pair's
      `upstream_ollama_tag`.
    - Otherwise → `None` (no auto-draft for this base).
    """
    if disabled:
        return None
    if override is not None:
        return override
    for pair in DRAFT_PAIRS:
        if pair.target_key == target_spec.key:
            return pair.upstream_ollama_tag
    return None


def validate_registry(base_models: dict[str, BaseModelSpec]) -> None:
    """Assert every DRAFT_PAIRS entry refers to compatible registry specs.

    Run at import time by `tests/unit/export/test_draft_registry.py`
    and by `scripts/refresh-registry.py`. Checks:

    - Both `target_key` and `draft_registry_key` exist in the
      registry.
    - They share `template` (same Go-template dialect).
    - They share `tokenizer_pre` (same BPE pre-tokenizer label —
      speculative decoding requires identical vocab).

    Raises `ValueError` on the first mismatch; the message names the
    offending pair so operators can fix it directly.
    """
    for pair in DRAFT_PAIRS:
        if pair.target_key not in base_models:
            raise ValueError(f"DRAFT_PAIRS: target_key {pair.target_key!r} not in BASE_MODELS")
        if pair.draft_registry_key not in base_models:
            raise ValueError(
                f"DRAFT_PAIRS: draft_registry_key {pair.draft_registry_key!r} "
                f"not in BASE_MODELS (paired with {pair.target_key!r})"
            )
        target = base_models[pair.target_key]
        draft = base_models[pair.draft_registry_key]
        if target.template != draft.template:
            raise ValueError(
                f"DRAFT_PAIRS: {pair.target_key!r} template {target.template!r} "
                f"!= {pair.draft_registry_key!r} template {draft.template!r}; "
                "speculative decoding requires identical chat templates"
            )
        if target.tokenizer_pre != draft.tokenizer_pre:
            raise ValueError(
                f"DRAFT_PAIRS: {pair.target_key!r} tokenizer_pre "
                f"{target.tokenizer_pre!r} != {pair.draft_registry_key!r} "
                f"tokenizer_pre {draft.tokenizer_pre!r}; speculative "
                "decoding requires an identical BPE pre-tokenizer"
            )
