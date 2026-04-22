"""Tokenizer load + fixup.

Three invariants enforced here (see CLAUDE.md pitfall #4):

1. **pad_token != eos_token.** HF defaults `pad_token` to `eos_token`
   on most bases; if `pad_token is None`, we MUST pick a different
   token, or labels get corrupted by mid-sequence EOS masking.
   Fallback order: `unk_token` → else add `<|pad|>` as a new special
   token (which grows the vocab and sets `tokenizer_grew=True` for
   the caller to propagate into the LoRA config).
2. **chat_template must be present.** Without it, SFTTrainer can't
   render `messages`-shaped rows. We surface a typed
   `TokenizerBringupError` rather than letting SFT fail deep inside
   TRL with an opaque message.
3. **Revision pinning.** Every load goes through the base model's
   40-char revision SHA — never a branch — so retrains under the
   same spec reproduce.

Returns a `TokenizerBringup` dataclass rather than a bare tokenizer so
the `tokenizer_grew` flag travels with the object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dlm.data.errors import TokenizerBringupError

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_PAD_TOKEN_LITERAL = "<|pad|>"


@dataclass(frozen=True)
class TokenizerBringup:
    """Result of `prepare_tokenizer`.

    `tokenizer_grew=True` means a new `<|pad|>` token was added to the
    vocab. The LoRA config MUST set
    `modules_to_save=["embed_tokens","lm_head"]` in that case —
    otherwise the new embedding row will not be trained and its
    output distribution is undefined.
    """

    tokenizer: PreTrainedTokenizerBase
    tokenizer_grew: bool
    pad_token: str
    chat_template: str


def prepare_tokenizer(hf_id: str, revision: str) -> TokenizerBringup:
    """Load the tokenizer for `hf_id` at `revision`, apply pad/template fixups."""
    from transformers import AutoTokenizer

    tok: Any = AutoTokenizer.from_pretrained(hf_id, revision=revision, use_fast=True)
    grew = _ensure_pad_token(tok)
    _ensure_chat_template(tok, hf_id=hf_id)

    pad = tok.pad_token
    chat_template = tok.chat_template
    assert isinstance(pad, str)
    assert isinstance(chat_template, str)

    return TokenizerBringup(
        tokenizer=tok,
        tokenizer_grew=grew,
        pad_token=pad,
        chat_template=chat_template,
    )


def _ensure_pad_token(tok: Any) -> bool:
    """Guarantee `tok.pad_token` is set AND distinct from `tok.eos_token`.

    Returns True iff a new special token was added to the vocab.
    """
    eos = getattr(tok, "eos_token", None)
    current_pad = getattr(tok, "pad_token", None)

    if current_pad is not None and current_pad != eos:
        return False

    # Either pad is unset, or it equals eos (the HF default we must override).
    unk = getattr(tok, "unk_token", None)
    if unk is not None and unk != eos:
        tok.pad_token = unk
        return False

    # Last resort: add a new pad token. This grows the vocab, which
    # forces training to update embed_tokens + lm_head.
    tok.add_special_tokens({"pad_token": _PAD_TOKEN_LITERAL})
    return True


def _ensure_chat_template(tok: Any, *, hf_id: str) -> None:
    template = getattr(tok, "chat_template", None)
    if template is None or not str(template).strip():
        raise TokenizerBringupError(
            f"base model {hf_id!r} has no chat_template; "
            "supply one via --chat-template or pick a registry base"
        )
