"""`ReplayStore` — high-level facade over `corpus.zst` + `index.json`.

Binds the low-level primitives (`corpus.append_snapshot`,
`index.load_index`, `sampler.sample`, `eviction.evict_until`) to a
concrete store path so callers don't juggle file paths themselves. The
store-level exclusive lock must be held for mutating operations —
this module doesn't acquire it, to avoid fighting the outer
training-run lifecycle.

Also provides `sample_rows()` — the glue that feeds
`build_dataset(..., replay_rows=...)` without the caller having to
understand snapshot → row shape herself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.replay.corpus import append_snapshot, iter_snapshots
from dlm.replay.index import load_index, save_index
from dlm.replay.models import IndexEntry, SectionSnapshot

if TYPE_CHECKING:
    import random
    from datetime import datetime

    from dlm.replay.sampler import Scheme

Row = dict[str, Any]


@dataclass(frozen=True)
class ReplayStore:
    """Facade bound to one store's `replay/` subdir.

    Construct via `ReplayStore.at(store_path.replay_corpus,
    store_path.replay_index)` — the path pair is kept explicit so the
    `StorePath` accessor remains the single source of truth for
    filesystem layout.
    """

    corpus_path: Path
    index_path: Path

    @classmethod
    def at(cls, corpus_path: Path, index_path: Path) -> ReplayStore:
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(corpus_path=corpus_path, index_path=index_path)

    # --- index ---------------------------------------------------------------

    def load(self) -> list[IndexEntry]:
        return load_index(self.index_path)

    def save(self, entries: list[IndexEntry]) -> None:
        save_index(self.index_path, entries)

    # --- corpus --------------------------------------------------------------

    def append(self, snapshot: SectionSnapshot) -> IndexEntry:
        """Append one snapshot, persist an updated index, return its entry.

        Index save happens on every append so a crash mid-training
        leaves the corpus + index consistent.

        **Performance:** each call does a full
        `load_index → append → save_index` cycle, which is O(n) in the
        existing index size. Fine for the one-shot append the trainer
        makes after each training cycle; **not** fine for loops like
        corpus imports or recovery flows. Use `append_many` whenever
        you have more than a handful of snapshots to add — the batch
        variant saves the index exactly once.
        """
        entry = append_snapshot(self.corpus_path, snapshot)
        self.save([*self.load(), entry])
        return entry

    def append_many(self, snapshots: list[SectionSnapshot]) -> list[IndexEntry]:
        """Batch variant: one index save at the end."""
        existing = self.load()
        new_entries = [append_snapshot(self.corpus_path, s) for s in snapshots]
        self.save([*existing, *new_entries])
        return new_entries

    # --- sampling → rows -----------------------------------------------------

    def sample_rows(
        self,
        *,
        k: int,
        now: datetime,
        rng: random.Random,
        scheme: Scheme = "recency",
    ) -> list[Row]:
        """Sample `k` snapshots and expand each to `sections_to_rows` dicts.

        A single INSTRUCTION snapshot can fan out to multiple rows (one
        per Q/A pair); same for PREFERENCE. The returned list is
        already flat — plug directly into
        `dlm.data.build_dataset(..., replay_rows=...)`.

        Each row's `_dlm_section_id` is prefixed with `replay:` and
        suffixed with the snapshot's `last_seen_at` timestamp. This
        prevents a rehydrated replay section from colliding with the
        same content in the current document under the splitter's
        `(seed, id, sub_index)` hash.
        """
        from dlm.replay.sampler import sample

        entries = self.load()
        picked = sample(entries, k=k, now=now, rng=rng, scheme=scheme)
        snapshots = list(iter_snapshots(self.corpus_path, picked))
        rows: list[Row] = []
        for snap in snapshots:
            rows.extend(_snapshot_to_rows(snap))
        return rows

    def sample_preference_rows(
        self,
        *,
        k: int,
        now: datetime,
        rng: random.Random,
        include_auto_mined: bool = True,
        scheme: Scheme = "recency",
    ) -> list[Row]:
        """Sample `k` *preference* snapshots; emit DPO-shaped rows.

        Mirrors `sample_rows` but pre-filters the index to
        preference-only entries before the weighted-reservoir draw.
        Falls back to an empty list when the corpus has no preference
        snapshots — callers at DPO-time decide whether zero replay is
        acceptable or not.

        `IndexEntry` doesn't carry `section_type` today, so we decode
        snapshots to partition. For the corpus sizes DLM realistically
        stores (<1k sections after eviction) the full decode is
        negligible compared to the training step itself.
        """
        from dlm.replay.sampler import sample

        entries = self.load()
        if not entries:
            return []

        snapshots = list(iter_snapshots(self.corpus_path, entries))
        preference_entries: list[IndexEntry] = []
        by_section_id: dict[str, SectionSnapshot] = {}
        for entry, snap in zip(entries, snapshots, strict=True):
            if snap.section_type != "preference":
                continue
            if not include_auto_mined and snap.auto_mined:
                continue
            preference_entries.append(entry)
            by_section_id[entry.section_id] = snap
        if not preference_entries:
            return []

        picked = sample(preference_entries, k=k, now=now, rng=rng, scheme=scheme)
        rows: list[Row] = []
        for entry in picked:
            snap = by_section_id[entry.section_id]
            rows.extend(_snapshot_to_rows(snap))
        return rows


def _snapshot_to_rows(snap: SectionSnapshot) -> list[Row]:
    """Expand one snapshot to its row-shape list.

    Mirrors `dlm.data.sections_to_rows._section_to_rows` but emits a
    replay-namespaced `_dlm_section_id` so replay rows don't collide
    with current-document rows of identical content.
    """
    replay_sid = f"replay:{snap.section_id}:{snap.last_seen_at.isoformat()}"

    if snap.section_type == "prose":
        text = snap.content.strip()
        if not text:
            return []
        return [{"text": text, "_dlm_section_id": replay_sid}]

    if snap.section_type == "instruction":
        from dlm.data.instruction_parser import parse_instruction_body

        pairs = parse_instruction_body(snap.content, section_id=snap.section_id)
        return [
            {
                "messages": [
                    {"role": "user", "content": pair.question},
                    {"role": "assistant", "content": pair.answer},
                ],
                "_dlm_section_id": replay_sid,
            }
            for pair in pairs
        ]

    # preference
    from dlm.data.preference_parser import parse_preference_body

    triples = parse_preference_body(snap.content, section_id=snap.section_id)
    return [
        {
            "prompt": t.prompt,
            "chosen": t.chosen,
            "rejected": t.rejected,
            "_dlm_section_id": replay_sid,
        }
        for t in triples
    ]
