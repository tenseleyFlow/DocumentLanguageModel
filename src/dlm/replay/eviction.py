"""Soft-cap eviction for `corpus.zst`.

The corpus grows forever by default. Once its total size exceeds the
user's `replay_budget_bytes`, eviction drops the oldest entries until
the size is back under cap — with two hard rules:

1. **Never evict a section belonging to the current document.** The
   caller passes `protect_ids` (typically every section_id in the
   parsed `.dlm`) and we refuse to drop any id in that set, even if
   that means the corpus stays over cap.
2. **Evict oldest first.** `added_at` ascending. Ties are broken by
   `section_id` for deterministic output.

Actually compacting `corpus.zst` is a pack/unpack concern — this
module only decides *which* index entries to drop. The caller updates
the index and, optionally, rewrites the corpus to reclaim the bytes.
(A sparse corpus with dead frames between live ones is a tolerable
intermediate state because frame-level random access only reads what
the index points at.)
"""

from __future__ import annotations

from collections.abc import Iterable

from dlm.replay.models import IndexEntry


def evict_until(
    entries: list[IndexEntry],
    *,
    max_bytes: int,
    protect_ids: Iterable[str] = (),
) -> tuple[list[IndexEntry], list[str]]:
    """Return `(kept_entries, evicted_ids)`.

    `kept_entries` is a fresh list in the original insertion order of
    the retained entries. `evicted_ids` is the list of dropped
    `section_id`s, in eviction order (oldest first).

    If `max_bytes == 0`, everything not in `protect_ids` is dropped —
    useful as a hard-reset path. Negative `max_bytes` is rejected.
    """
    if max_bytes < 0:
        raise ValueError(f"max_bytes must be >= 0, got {max_bytes}")

    protect_set = frozenset(protect_ids)
    total_bytes = sum(e.length for e in entries)
    if total_bytes <= max_bytes:
        return list(entries), []

    # Oldest-first eviction ordering; ties broken by section_id.
    eviction_candidates = sorted(
        (e for e in entries if e.section_id not in protect_set),
        key=lambda e: (e.added_at, e.section_id),
    )

    dropped_ids: set[str] = set()
    evicted_order: list[str] = []
    current = total_bytes
    for entry in eviction_candidates:
        if current <= max_bytes:
            break
        dropped_ids.add(entry.section_id)
        evicted_order.append(entry.section_id)
        current -= entry.length

    kept = [e for e in entries if e.section_id not in dropped_ids]
    return kept, evicted_order
