"""Canonical tokenizer-vocabulary-extension contract (Sprint 12b, audit F02/F06).

A training run whose bringup (Sprint 07) adds a new special token grows
the vocabulary. Every downstream stage — LoRA config (`modules_to_save`),
export preflight (`tokenizer_from_adapter_dir.vocab_size ==
gguf_base.vocab_size + N_added`), Modelfile stops (Sprint 12) — depends
on *the same* predicate for "did this tokenizer grow". This module is
that predicate's canonical home.

Two functions:

- `tokenizer_grew(base, final)` — True iff `vocab_size` changed or the
  added-token set changed. Works for any `PreTrainedTokenizerBase`
  (BPE or SentencePiece family).
- `modules_to_save_for_growth(grew)` — `["embed_tokens", "lm_head"]`
  when `grew=True`, else `[]`. Sprint 09 calls this when building the
  LoRA config. Per pitfall #4, without the modules_to_save entry the
  new embedding row's output is undefined.

The predicate lives in `dlm.data` because the tokenizer itself is a
data-stage concern; the downstream callers (`dlm.train.adapter`,
`dlm.export.preflight`) import from here rather than reimplementing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def tokenizer_grew(base: PreTrainedTokenizerBase, final: PreTrainedTokenizerBase) -> bool:
    """True iff `final` has a larger vocab or different added-token set than `base`.

    `vocab_size` comparison catches the `add_special_tokens` path used by
    Sprint 07's pad fallback. The `get_added_vocab()` set-comparison
    catches cases where an added token was *replaced* with a same-count
    variant (vocab size unchanged but contents differ) — rare but
    possible when users manually mutate the tokenizer between runs.
    """
    if final.vocab_size != base.vocab_size:
        return True
    base_added = set(base.get_added_vocab())
    final_added = set(final.get_added_vocab())
    return base_added != final_added


def modules_to_save_for_growth(grew: bool) -> list[str]:
    """Return the LoRA `modules_to_save` list for a grown tokenizer.

    `["embed_tokens", "lm_head"]` forces the LoRA config to train the
    input embeddings and the output projection alongside the adapter
    ranks, so the new token(s) have meaningful outputs. `[]` on an
    unchanged tokenizer keeps the adapter small.
    """
    return ["embed_tokens", "lm_head"] if grew else []
