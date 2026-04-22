"""Replay corpus — rolling append-only zstd+CBOR store of section snapshots.

Public surface:

- `ReplayStore` — facade bound to a store's `replay/` subdir.
- `SectionSnapshot`, `IndexEntry` — records stored in `corpus.zst` and
  `index.json` respectively.
- `ChangeSet`, `diff_against_manifest` — content-delta against the
  previous training run.
- `sample`, `evict_until` — low-level sampling + eviction primitives.
- Typed errors: `ReplayError`, `CorpusCorruptError`,
  `IndexCorruptError`, `SamplerError`.
"""

from __future__ import annotations

from dlm.replay.delta import ChangeSet, diff_against_manifest
from dlm.replay.errors import (
    CorpusCorruptError,
    IndexCorruptError,
    ReplayError,
    SamplerError,
)
from dlm.replay.eviction import evict_until
from dlm.replay.models import IndexEntry, SectionSnapshot
from dlm.replay.sampler import sample
from dlm.replay.store import ReplayStore

__all__ = [
    "ChangeSet",
    "CorpusCorruptError",
    "IndexCorruptError",
    "IndexEntry",
    "ReplayError",
    "ReplayStore",
    "SamplerError",
    "SectionSnapshot",
    "diff_against_manifest",
    "evict_until",
    "sample",
]
