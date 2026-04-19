"""Perplexity utility.

Perplexity of a held-out validation split is `exp(cross_entropy_loss)`.
Lower is better; a language model that assigns probability 1 to every
token has loss 0 and perplexity 1. On small documents the numbers are
noisy — the val set is rarely large enough for a stable PPL — but the
*trend* across eval steps is the signal.

Pulled out as its own module so `dlm metrics` (Sprint 20) can import
it without pulling in torch / transformers.
"""

from __future__ import annotations

import math


def perplexity(loss: float) -> float:
    """Return `exp(loss)`, or `math.inf` for non-finite / negative inputs.

    A non-finite loss (NaN / inf) would cause `math.exp` to overflow or
    return NaN; we surface `math.inf` so log / metric pipelines have a
    sortable sentinel rather than a bad float.
    """
    if not math.isfinite(loss) or loss < 0.0:
        return math.inf
    try:
        return math.exp(loss)
    except OverflowError:
        return math.inf
