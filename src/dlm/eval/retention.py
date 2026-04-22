"""Retention metric — eval on a fixed slice of the replay corpus.

At every eval step the trainer computes val loss on the current
document's held-out split. That tells us how well the model fits
recent content. It does NOT tell us whether the model still
remembers what it learned two retrains ago — the "catastrophic
forgetting" failure mode the replay corpus was designed to prevent.

This module picks a **stable 5% slice** of the replay corpus at run
start, reserves it as eval-only (the trainer never sees it), and
reports loss on that slice alongside val loss. A retention_delta >>
val_delta between runs is the forgetting signal the UI surfaces.

Design:
- `build_retention_slice(replay_store, *, frac, seed)` returns a list
  of `IndexEntry` the caller can rehydrate into rows. Seed-stable:
  same corpus + seed → same slice → same "held-out" across runs.
- The slice is disjoint from the training sample (the sampler draws
  from the remainder of the corpus; trainer threads the retention
  entries' ids to `exclude`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from dlm.eval.errors import RetentionSliceError
from dlm.replay.models import IndexEntry

_DEFAULT_FRAC = 0.05


@dataclass(frozen=True)
class RetentionSlice:
    """Seed-stable eval-only slice of the replay corpus."""

    entries: list[IndexEntry]
    seed: int
    frac: float

    @property
    def section_ids(self) -> set[str]:
        return {e.section_id for e in self.entries}


def build_retention_slice(
    entries: list[IndexEntry],
    *,
    frac: float = _DEFAULT_FRAC,
    seed: int = 0,
) -> RetentionSlice:
    """Pick a `frac` fraction of `entries` to reserve as retention-only.

    Entries are hashed against `(seed, section_id)` and the top-k by
    hash are selected. This is stable — adding a section to the corpus
    doesn't reshuffle what's already been designated retention — and
    deterministic under the same seed.

    Raises `RetentionSliceError` on empty input or a frac outside (0, 1).
    """
    if not 0.0 < frac < 1.0:
        raise RetentionSliceError(f"frac must be in (0, 1), got {frac!r}")
    if not entries:
        raise RetentionSliceError("cannot build retention slice from empty corpus")

    # Always pick at least one; round up so small corpora still have
    # a retention signal.
    k = max(1, int(len(entries) * frac + 0.999))
    keyed = sorted(entries, key=lambda e: _retention_key(e.section_id, seed))
    picked = keyed[:k]
    return RetentionSlice(entries=picked, seed=seed, frac=frac)


def _retention_key(section_id: str, seed: int) -> str:
    h = hashlib.sha256(f"{seed}\x00{section_id}".encode())
    return h.hexdigest()


def retention_delta(
    *,
    current_retention_loss: float | None,
    previous_retention_loss: float | None,
) -> float | None:
    """`current - previous`; None if either side is missing.

    Reported in the `TrainingSummary.retention_loss_delta` field. A
    positive delta means retention loss went UP — the model is
    forgetting. The magnitude relative to `final_val_loss` is the
    honest forgetting signal; a rising retention loss alongside falling
    val loss is the canonical catastrophic-forgetting fingerprint.
    """
    if current_retention_loss is None or previous_retention_loss is None:
        return None
    return current_retention_loss - previous_retention_loss
