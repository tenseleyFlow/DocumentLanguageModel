"""Vocabulary-fit report for the base tokenizer against the doc corpus.

Run at training start so the user can spot a mismatched tokenizer
before burning an hour on a train that won't converge. The report is
descriptive, not prescriptive — we never auto-swap or auto-extend
the tokenizer (that would quietly break the base model's alignment);
we just surface the numbers that predict "your base tokenizer is a
bad fit for this domain."

Three signals:

1. **tokens-per-word** (TPW): total subword tokens divided by
   whitespace-delimited word count. A number close to 1.0 means words
   survive intact; 2.0+ means the tokenizer is aggressively splitting
   (e.g., code tokens on a prose-tuned base).
2. **top-N frequent tokens**: the most-used subwords and their
   decoded surface form. Useful for spotting weird fragment patterns
   (e.g., `Ġd`, `ata` as frequent merges → `data` would be one
   efficient token if the base had seen it).
3. **unk_hits**: count of `<unk>` tokens in the tokenized corpus.
   Should be zero for modern BPE. Non-zero ≈ the tokenizer has
   rare-character holes (emoji, some unicode scripts) and the user
   should consider a different base.

Pure-python: no torch, no GPU. Fast enough to run synchronously before
the first training step on docs up to a few hundred KB.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VocabGapReport:
    """Per-corpus tokenizer-fit summary.

    - `total_tokens` / `total_words` — the raw counts behind TPW.
    - `tokens_per_word` — total_tokens / total_words (0.0 if no words).
    - `unk_hits` — count of ``<unk>`` tokens; 0 on healthy BPE.
    - `top_tokens` — list of (decoded_surface, frequency) pairs,
      descending, capped at the report's ``top_n`` parameter.
    - `has_unk` — boolean sugar over `unk_hits > 0`.
    """

    total_tokens: int
    total_words: int
    tokens_per_word: float
    unk_hits: int
    top_tokens: list[tuple[str, int]]

    @property
    def has_unk(self) -> bool:
        return self.unk_hits > 0


_WORD_RE = re.compile(r"\b\w+\b")


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def compute_vocab_gap(
    token_ids: list[int],
    *,
    text: str,
    unk_token_id: int | None,
    decoded_tokens: list[str],
    top_n: int = 10,
) -> VocabGapReport:
    """Compute the report from already-tokenized inputs.

    Separated from `report()` so unit tests can drive it without a
    real tokenizer. `decoded_tokens` must be aligned with `token_ids`
    (same length, same order); it holds the surface form of each ID
    for the top-N display. `text` is the original corpus used only
    for the word count.

    `unk_token_id` may be `None` if the tokenizer doesn't define one;
    in that case `unk_hits` is reported as 0.
    """
    if len(token_ids) != len(decoded_tokens):
        raise ValueError(
            f"token_ids/decoded_tokens length mismatch: "
            f"{len(token_ids)} vs {len(decoded_tokens)}"
        )
    if top_n < 0:
        raise ValueError(f"top_n must be non-negative, got {top_n}")

    total_tokens = len(token_ids)
    total_words = _count_words(text)
    tpw = total_tokens / total_words if total_words else 0.0

    unk_hits = (
        sum(1 for tid in token_ids if tid == unk_token_id)
        if unk_token_id is not None
        else 0
    )

    counts: Counter[str] = Counter(decoded_tokens)
    top_tokens = counts.most_common(top_n)

    return VocabGapReport(
        total_tokens=total_tokens,
        total_words=total_words,
        tokens_per_word=tpw,
        unk_hits=unk_hits,
        top_tokens=top_tokens,
    )


def report(text: str, tokenizer: Any, *, top_n: int = 10) -> VocabGapReport:  # pragma: no cover - network/heavy
    """Run the base tokenizer over `text` and compute the fit report.

    Heavy-import shell around `compute_vocab_gap` — covered by the slow
    integration test. The kept-pure split means `compute_vocab_gap`
    carries the coverage weight in the unit suite.
    """
    encoded = tokenizer(text, add_special_tokens=False)
    ids: list[int] = list(encoded["input_ids"])
    decoded: list[str] = tokenizer.convert_ids_to_tokens(ids)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    return compute_vocab_gap(
        ids,
        text=text,
        unk_token_id=unk_id,
        decoded_tokens=decoded,
        top_n=top_n,
    )


def render_report(r: VocabGapReport) -> str:
    """One-screen plain-text rendering of the report for the training log.

    Three sections: headline ratios, top-N tokens, `<unk>` warning if
    any hits. Every line is ≤80 columns; the block suits a terminal
    log tail.
    """
    lines: list[str] = [
        "vocabulary gap report",
        f"  tokens per word : {r.tokens_per_word:.2f} "
        f"({r.total_tokens} tokens / {r.total_words} words)",
        f"  <unk> hits      : {r.unk_hits}",
    ]
    if r.has_unk:
        lines.append(
            "  WARNING: non-zero <unk> count — tokenizer has rare-character"
        )
        lines.append(
            "  holes for this domain. Consider a different base model."
        )
    if r.top_tokens:
        lines.append("  top tokens:")
        width = max(len(t) for t, _ in r.top_tokens)
        for tok, freq in r.top_tokens:
            lines.append(f"    {tok.ljust(width)}  {freq}")
    return "\n".join(lines)
