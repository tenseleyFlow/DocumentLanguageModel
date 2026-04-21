"""Pre-tokenization pass that feeds SFTTrainer a cached, pre-processed
dataset.

TRL 1.2's ``SFTTrainer._prepare_dataset`` skips its own chat-template /
EOS / tokenize pipeline when the incoming dataset already contains an
``input_ids`` column. ``pretokenize_rows`` produces exactly that shape
and consults ``TokenizedCache`` per-row — a cache hit skips the
``tokenizer()`` call, which is the dominant cost on 50K-file directive
corpora.

Tokenization must be bit-identical to what TRL would have emitted on
its own. TRL's ``_tokenize`` dispatches:

- ``messages`` (list[dict]) → ``apply_chat_template(messages,
  tokenize=True, return_dict=True)``
- ``text`` (str) → ``tokenizer(text=text)`` after appending
  ``tokenizer.eos_token`` (so loss sees the stop signal)

``pretokenize_rows`` replicates both paths and strips the internal
``_dlm_section_id`` field before handing rows to TRL.

Labels are *not* cached. In our current SFT shape
(``completion_only_loss=None``, default collator), labels are derived
from ``input_ids`` by the collator at batch time — caching them is
redundant and would only matter if the `.dlm` opts into
``completion_only_loss=True`` later.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from dlm.directives.cache import CachedTokens
from dlm.directives.cache_key import CacheKey, tokenizer_sha256

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from dlm.directives.cache import TokenizedCache


_LOG = logging.getLogger(__name__)

_SECTION_ID_KEY = "_dlm_section_id"


@dataclass(frozen=True)
class TokenizationStats:
    """Aggregate counts for a single pre-tokenize pass.

    Consumed by ``MetricsRecorder.record_tokenization`` at run end so
    ``dlm metrics`` surfaces per-run hit rate alongside the existing
    step/eval drill-downs.
    """

    total_sections: int
    cache_hits: int
    cache_misses: int
    total_tokenize_seconds: float
    cache_bytes_after: int


def pretokenize_rows(
    rows: list[dict[str, Any]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    sequence_len: int,
    cache: TokenizedCache | None = None,
) -> tuple[list[dict[str, Any]], TokenizationStats]:
    """Tokenize each row, consulting ``cache`` when provided.

    Input rows carry ``messages`` or ``text`` plus a ``_dlm_section_id``
    metadata field (attached by ``sections_to_rows``). Output rows
    carry ``input_ids`` (list[int]) and ``attention_mask`` (list[int]) —
    enough for TRL's ``is_processed`` check to skip its own
    tokenization. The ``_dlm_section_id`` field is preserved so the
    splitter and downstream callbacks can key on section identity.

    When ``cache is None`` every row tokenizes fresh; when supplied,
    misses tokenize then populate. The returned ``TokenizationStats``
    sum over all rows regardless of cache state.

    Rows that carry neither ``messages`` nor ``text`` — preference
    rows destined for DPOTrainer — pass through untouched. DPOTrainer
    owns its own tokenization path (Sprint 17).
    """
    sha = tokenizer_sha256(tokenizer) if cache is not None else ""
    stats_hits = 0
    stats_misses = 0
    tokenize_seconds = 0.0

    out: list[dict[str, Any]] = []
    for row in rows:
        # Preference rows skip pre-tokenization — they'll be consumed by
        # DPOTrainer / ORPOTrainer, which have their own tokenizers.
        if _is_preference_row(row):
            out.append(dict(row))
            continue

        section_id = row.get(_SECTION_ID_KEY)
        if not isinstance(section_id, str):
            # No section id → can't cache. Happens for synthetic rows in
            # tests and for the occasional non-directive row that
            # somehow dropped the metadata field.
            tokens, t = _tokenize_row(row, tokenizer)
            tokenize_seconds += t
            stats_misses += 1
            out.append(_row_with_tokens(row, tokens))
            continue

        key = CacheKey(section_id=section_id, tokenizer_sha=sha, sequence_len=sequence_len)
        hit = cache.get(key) if cache is not None else None
        if hit is not None:
            stats_hits += 1
            tokens = hit
        else:
            raw, t = _tokenize_row(row, tokenizer)
            tokenize_seconds += t
            stats_misses += 1
            tokens = raw
            if cache is not None:
                cache.put(key, tokens)

        out.append(_row_with_tokens(row, tokens))

    cache_bytes = cache.total_bytes if cache is not None else 0
    stats = TokenizationStats(
        total_sections=len(rows),
        cache_hits=stats_hits,
        cache_misses=stats_misses,
        total_tokenize_seconds=tokenize_seconds,
        cache_bytes_after=cache_bytes,
    )
    return out, stats


def _is_preference_row(row: dict[str, Any]) -> bool:
    return (
        row.get("prompt") is not None
        and row.get("chosen") is not None
        and row.get("rejected") is not None
    )


def _tokenize_row(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[CachedTokens, float]:
    """Tokenize one row and return ``(CachedTokens, elapsed_seconds)``.

    Mirrors TRL ``SFTTrainer._tokenize``:

    - ``messages``: ``apply_chat_template(..., tokenize=True,
      return_dict=True)``. No extra EOS — the template usually emits
      one itself.
    - ``text``: append ``tokenizer.eos_token`` (unless already there)
      then tokenize. Matches TRL's ``add_eos`` map step + ``_tokenize``.
    """
    t0 = time.perf_counter()

    messages = row.get("messages")
    if messages is not None:
        # `apply_chat_template(tokenize=True, return_dict=True)` always
        # returns a dict-like BatchEncoding; the union type in HF's
        # stubs includes string / list return shapes for the other
        # flag combinations. Narrow for mypy.
        raw: Any = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
        )
        input_ids = _as_int_list(raw["input_ids"])
        attention_mask = _as_int_list(raw.get("attention_mask", [1] * len(input_ids)))
    else:
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"pretokenize_rows: row has neither `messages` nor `text`: "
                f"keys={sorted(row.keys())}"
            )
        eos = tokenizer.eos_token
        if eos is not None and not text.endswith(eos):
            text = text + eos
        raw = tokenizer(text=text)
        input_ids = _as_int_list(raw["input_ids"])
        attention_mask = _as_int_list(raw.get("attention_mask", [1] * len(input_ids)))

    tokens = CachedTokens(
        input_ids=np.asarray(input_ids, dtype=np.int64),
        attention_mask=np.asarray(attention_mask, dtype=np.int64),
    )
    return tokens, time.perf_counter() - t0


def _as_int_list(obj: Any) -> list[int]:
    """Coerce tokenizer output (list, np.ndarray, torch.Tensor) to list[int]."""
    if isinstance(obj, list):
        return [int(x) for x in obj]
    # numpy / torch: both expose `tolist()` returning nested lists.
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        flat = tolist()
        # Handle the rare case of a 2D batch-of-1 slipping through.
        if flat and isinstance(flat[0], list):
            flat = flat[0]
        return [int(x) for x in flat]
    return [int(x) for x in obj]


def _row_with_tokens(row: dict[str, Any], tokens: CachedTokens) -> dict[str, Any]:
    """Reshape a row into the TRL-ready form.

    Drops ``messages`` / ``text`` (TRL's ``is_processed`` check
    ignores them anyway) and preserves ``_dlm_section_id`` so the
    deterministic train/val splitter can still key on it.
    """
    out: dict[str, Any] = {
        "input_ids": tokens.input_ids.tolist(),
        "attention_mask": tokens.attention_mask.tolist(),
    }
    sid = row.get(_SECTION_ID_KEY)
    if isinstance(sid, str):
        out[_SECTION_ID_KEY] = sid
    return out
