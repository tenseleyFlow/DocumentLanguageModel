"""Cache key derivation for the tokenized-section cache.

A `CacheKey` identifies one tokenization of one section under one
tokenizer at one sequence length. All three inputs participate in the
filename AND the manifest key — a change to any invalidates the entry.

`tokenizer_sha256` computes the canonical fingerprint. The fast-
tokenizer path reads `tokenizer.json` bytes directly; the legacy
SentencePiece path falls back to a deterministic dump of
`__getstate__()`. Both paths pin the result on the tokenizer
instance so repeated calls in one run cost one hash.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_FINGERPRINT_ATTR = "_dlm_tokenizer_sha256"


@dataclass(frozen=True)
class CacheKey:
    """Composite key for one tokenized entry.

    `section_id` is 16-hex from `Section.section_id`. `tokenizer_sha`
    is the 64-hex sha256 from `tokenizer_sha256`. `sequence_len` is
    the tokenizer's max sequence length for this run.
    """

    section_id: str
    tokenizer_sha: str
    sequence_len: int

    def as_filename(self) -> str:
        """Stable shard/name for on-disk storage.

        Format: `<section_id>.<tok_sha[:12]>.seq<sequence_len>.npz`.
        The 12-char tokenizer-sha prefix (48 bits) is plenty to avoid
        collisions within a cache: caches are per-store, typical stores
        see O(1) tokenizer families (one pinned fingerprint per base
        model), so the collision space is "one entry per section per
        tokenizer family" — astronomically far from the 2^24-entry
        birthday threshold. The full sha is persisted in the manifest
        for verification if a collision ever occurs in practice.
        """
        return f"{self.section_id}.{self.tokenizer_sha[:12]}.seq{self.sequence_len}.npz"

    def shard(self) -> str:
        """First 2 hex chars of section_id — the directory shard."""
        return self.section_id[:2]


def tokenizer_sha256(tokenizer: Any) -> str:
    """Canonical sha256 of a HuggingFace tokenizer's bytes.

    For fast tokenizers (`tokenizer.backend_tokenizer` is a
    `tokenizers.Tokenizer`), serialize via `to_str()` — the canonical
    JSON form includes vocab, merges, normalizer, pre-tokenizer,
    post-processor, and added-tokens. A bump in any of those shifts
    the sha and invalidates the cache, which is exactly what we want.

    For legacy SentencePiece-only tokenizers (no backend), fall back
    to a deterministic `json.dumps(sorted dict)` of the vocab + special
    tokens + model_max_length. This is weaker but deterministic
    enough for our usage pattern (caches are per-store, not shared).

    The result is pinned on the tokenizer instance via a private
    attribute so repeated calls in one run are O(1).
    """
    pinned: str | None = getattr(tokenizer, _FINGERPRINT_ATTR, None)
    if pinned is not None:
        return pinned

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        try:
            canonical = backend.to_str()
        except Exception:  # noqa: BLE001 — defensive fallback
            canonical = _legacy_canonical(tokenizer)
    else:
        canonical = _legacy_canonical(tokenizer)

    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # Some tokenizer classes forbid new attributes; fine, just re-hash
    # next call.
    with contextlib.suppress(AttributeError, TypeError):
        object.__setattr__(tokenizer, _FINGERPRINT_ATTR, sha)
    return sha


def _legacy_canonical(tokenizer: PreTrainedTokenizerBase) -> str:
    """Legacy fallback: deterministic JSON dump of the tokenizer's
    identity-bearing state.

    Keys are sorted so the hash is stable across Python dict-ordering
    quirks. Non-serializable values are stringified.
    """
    state: dict[str, object] = {
        "vocab_size": getattr(tokenizer, "vocab_size", 0),
        "model_max_length": getattr(tokenizer, "model_max_length", 0),
        "pad_token": str(getattr(tokenizer, "pad_token", "")),
        "eos_token": str(getattr(tokenizer, "eos_token", "")),
        "bos_token": str(getattr(tokenizer, "bos_token", "")),
        "unk_token": str(getattr(tokenizer, "unk_token", "")),
        "cls_token": str(getattr(tokenizer, "cls_token", "")),
        "sep_token": str(getattr(tokenizer, "sep_token", "")),
        "mask_token": str(getattr(tokenizer, "mask_token", "")),
        "added_tokens_count": len(getattr(tokenizer, "added_tokens_decoder", {}) or {}),
        "class": tokenizer.__class__.__name__,
    }
    return json.dumps(state, sort_keys=True, default=str)
