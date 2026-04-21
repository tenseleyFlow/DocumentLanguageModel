"""Tag-weighted row expansion — deterministic row repetition.

Operators declare `weights: {tag_key: {tag_value: float}}` in a
`.dlm/training.yaml` to up- or down-scale how often rows with that
tag appear in the training corpus. We implement it as *row
repetition* rather than per-row loss scaling:

- weight = 1.0  → row appears once (no-op)
- weight = 0.0  → row dropped
- weight = 2.0  → row appears twice
- weight = 2.5  → row appears twice, plus a deterministic 50%
                  chance of a third copy (seeded by section_id)
- weight = 0.5  → row appears with deterministic 50% keep probability

Multiple tag keys compose multiplicatively: a row tagged
`{docstring: true, generated: true}` with
`{docstring: {true: 2.0}, generated: {true: 0.5}}` ends up at
weight 1.0 (= 2.0 × 0.5).

Determinism: the keep/extra-copy decision is a hash of
`(seed, section_id, fractional_index)`. Same seed + same corpus →
same expanded row list, bit-exact. This preserves the Sprint 31.5
determinism guarantee: a cached run and an uncached run on the same
weights config produce byte-identical adapter weights.

**Why row repetition, not per-row loss scaling?** Sprint 31.5's
hard-won bit-identity against TRL's `_tokenize` would be lost the
moment we subclassed `SFTTrainer.compute_loss` to multiply by a
sample-weights tensor — any TRL internal refactor of the loss path
becomes a silent correctness bug. Expansion is a dataset-level
transform; every downstream layer (pretokenize cache, TRL
collator, AdamW) sees a plain list of rows and stays dumb.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

Row = dict[str, Any]
WeightsMap = Mapping[str, Mapping[str, float]]


def resolve_row_weight(row_tags: Mapping[str, str], weights: WeightsMap) -> float:
    """Compose the effective weight for a row from its tags + weights map.

    Missing tag keys and unmatched tag values contribute 1.0 (no
    scaling). Matching `(tag_key, tag_value)` entries multiply in.
    Order-independent.
    """
    weight = 1.0
    for tag_key, tag_value in row_tags.items():
        inner = weights.get(tag_key)
        if inner is None:
            continue
        scale = inner.get(tag_value)
        if scale is None:
            continue
        weight *= scale
    return weight


def _keep_fraction(section_id: str, seed: int, fractional: float) -> bool:
    """Deterministic Bernoulli: True with probability `fractional`.

    Uses BLAKE2b over `(seed, section_id)` — cheap, collision-
    resistant, and reproducible across platforms. The section_id is
    stable under the content-addressed store, so the keep/drop
    decision for a given row depends only on seed + content, never
    on row position.
    """
    if fractional <= 0.0:
        return False
    if fractional >= 1.0:
        return True
    h = hashlib.blake2b(f"{seed}:{section_id}".encode(), digest_size=8).digest()
    # Map the first 8 bytes to [0, 1) — integer / 2**64.
    roll = int.from_bytes(h, "big") / float(1 << 64)
    return roll < fractional


def expand_rows_by_weight(
    rows: Sequence[Row],
    weights: WeightsMap,
    *,
    seed: int,
) -> list[Row]:
    """Return a new row list where each input row is repeated (or dropped)
    per its composed weight.

    A row without a `_dlm_row_tags` key gets weight 1.0 (untouched).
    An empty `weights` map is a no-op (returns a shallow copy of
    `rows`). Section-ID preservation means the replay corpus still
    tracks per-row identity — the N copies of a repeated row share
    a section_id, which matches the Sprint 08 semantics of "retraining
    on the same content N times".
    """
    if not weights:
        return list(rows)

    expanded: list[Row] = []
    for row in rows:
        row_tags = row.get("_dlm_row_tags") or {}
        weight = resolve_row_weight(row_tags, weights)
        if weight <= 0.0:
            continue
        integer_copies = int(weight)
        fractional = weight - integer_copies
        for _ in range(integer_copies):
            expanded.append(row)
        if fractional > 0.0:
            section_id = str(row.get("_dlm_section_id", ""))
            if _keep_fraction(section_id, seed, fractional):
                expanded.append(row)
    return expanded


def weight_distribution(
    rows: Sequence[Row],
) -> dict[str, dict[str, int]]:
    """Count original rows per `(tag_key, tag_value)` for summary reporting.

    Takes the pre-expansion row list so users can audit how many rows
    were candidates for each rule, independent of how many copies
    the expansion produced.
    """
    dist: dict[str, dict[str, int]] = {}
    for row in rows:
        row_tags = row.get("_dlm_row_tags") or {}
        for tag_key, tag_value in row_tags.items():
            inner = dist.setdefault(tag_key, {})
            inner[tag_value] = inner.get(tag_value, 0) + 1
    return dist
