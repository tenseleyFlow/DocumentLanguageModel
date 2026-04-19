"""Sample `k` index entries from the corpus for replay training.

Two schemes:

- `"recency"` (default): weight = `1 / (1 + age_days)` + a small floor so
  very old sections remain sampleable. Newer sections dominate but the
  distribution never collapses to a single section.
- `"uniform"`: every entry has equal weight; useful for tests and for
  documents whose content churn is mostly on stable prose.

Determinism
-----------

Sampling is fully deterministic given a `random.Random` instance. The
caller seeds it from `run_seed + manifest.adapter_version` so two
identical training runs draw the same replay set (CI regression check).

We use the stdlib `random` module rather than `numpy.random` on
purpose: stdlib is already a runtime dep, and the algorithm here
(weighted reservoir via the exponential-key trick) is trivial enough
that the numpy dependency for this one module isn't worth it.

Algorithm
---------

`_weighted_reservoir(entries, k, weights, rng)` implements the
A-ExpJ algorithm (Efraimidis & Spirakis, 2006):

    key_i = rng.random() ** (1 / w_i)
    select the k entries with the largest keys

This is an optimal single-pass reservoir sampler for weighted entries
with `k ≪ n`. Reading all entries is fine — the index is tiny relative
to the corpus body.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Literal

from dlm.replay.errors import SamplerError
from dlm.replay.models import IndexEntry

Scheme = Literal["recency", "uniform"]

_RECENCY_WEIGHT_FLOOR = 0.01


def sample(
    entries: list[IndexEntry],
    *,
    k: int,
    now: datetime,
    rng: random.Random,
    scheme: Scheme = "recency",
) -> list[IndexEntry]:
    """Return `k` entries drawn from `entries` according to `scheme`.

    If `k >= len(entries)`, returns a stable-ordered copy of the whole
    input (sorted by section_id) — a full sweep, no sampling. This
    matches the intuition that asking for more samples than the corpus
    has is a non-failure "give me everything" request.
    """
    if k < 0:
        raise SamplerError(f"k must be non-negative, got {k}")
    if k == 0:
        return []

    # Always iterate in a stable order so the sampler depends only on
    # the RNG — not on filesystem / dict enumeration order.
    ordered = sorted(entries, key=lambda e: e.section_id)
    if k >= len(ordered):
        return ordered

    weights = _compute_weights(ordered, now=now, scheme=scheme)
    return _weighted_reservoir(ordered, weights=weights, k=k, rng=rng)


def _compute_weights(entries: list[IndexEntry], *, now: datetime, scheme: Scheme) -> list[float]:
    if scheme == "uniform":
        return [1.0] * len(entries)
    if scheme == "recency":
        return [_recency_weight(entry, now=now) for entry in entries]
    raise SamplerError(f"unknown scheme: {scheme!r}")


def _recency_weight(entry: IndexEntry, *, now: datetime) -> float:
    """`1 / (1 + age_days)`, floored so ancient sections stay sampleable.

    `entry.weight` overrides the recency calc — set it manually to
    up- or down-weight a specific section. Defaults to `1.0`, which
    leaves the pure recency curve in place.
    """
    age = max(now - entry.added_at, timedelta(0))
    age_days = age.total_seconds() / 86_400.0
    base = 1.0 / (1.0 + age_days)
    return max(entry.weight * base, _RECENCY_WEIGHT_FLOOR)


def _weighted_reservoir(
    entries: list[IndexEntry],
    *,
    weights: list[float],
    k: int,
    rng: random.Random,
) -> list[IndexEntry]:
    """A-ExpJ: key = `u^(1/w)`, keep the top-k."""
    keyed: list[tuple[float, int]] = []  # (key, stable_index) for ties
    for i, w in enumerate(weights):
        if w <= 0:
            # Zero-weight entries never get sampled. Skip to avoid a
            # ZeroDivisionError from the `1 / w` exponent.
            continue
        u = rng.random()
        # Guard against u == 0.0 (math.log(0) is -inf) — retry once.
        if u == 0.0:
            u = rng.random() or 1e-300
        key = math.log(u) / w
        keyed.append((key, i))

    # Largest key wins — keep top-k via partial sort.
    keyed.sort(reverse=True)
    return [entries[i] for _, i in keyed[:k]]
