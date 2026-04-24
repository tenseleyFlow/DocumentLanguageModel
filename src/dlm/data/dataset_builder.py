"""End-to-end: parsed `.dlm` sections → (train_ds, val_ds).

This is the single entry point the trainer calls. It:

1. Flattens `sections` to dict rows via `sections_to_rows`.
2. Optionally concatenates a replay-corpus row iterable (we just
   accept an iterable here to keep the dependency one-directional).
3. Splits into train / val via the deterministic splitter.

The split is keyed on each row's `_dlm_section_id` + sub-index, so
replay rows must also carry a stable `_dlm_section_id` — the corpus
reader stamps one derived from the originating document's version.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from dlm.data.sections_to_rows import sections_to_rows
from dlm.data.splitter import split
from dlm.data.weighted_rows import expand_rows_by_weight
from dlm.doc.sections import Section

if TYPE_CHECKING:
    from datasets import Dataset

    from dlm.store.blobs import BlobStore

Row = dict[str, Any]


def build_dataset(
    sections: list[Section],
    *,
    val_frac: float = 0.1,
    seed: int,
    replay_rows: Iterable[Row] | None = None,
    weights: Mapping[str, Mapping[str, float]] | None = None,
    blob_store: BlobStore | None = None,
    image_token: str = "<image>",
    audio_token: str = "<|AUDIO|>",
) -> tuple[Dataset, Dataset]:
    """Build a (train, val) `Dataset` pair from parsed `.dlm` sections.

    `seed` is required (not defaulted) so the split is always traceable
    to a manifest entry; `val_frac=0.1` matches the current default.

    `weights`, when non-empty, expands rows by `(tag_key, tag_value)`
    multipliers before the train/val split — integer factors duplicate
    rows, fractional factors drive a deterministic per-section keep
    decision. The expansion applies to both in-document and replay
    rows so retention behaves uniformly.

    `blob_store` + `image_token` + `audio_token` flow through to
    `sections_to_rows` for media-section emission. Callers with
    vision-language or audio-language bases must supply the store;
    text-only documents leave the defaults.
    """
    rows = sections_to_rows(
        sections,
        blob_store=blob_store,
        image_token=image_token,
        audio_token=audio_token,
    )
    if replay_rows is not None:
        rows.extend(r for r in replay_rows if not _is_preference_row(r))

    if not rows:
        raise ValueError(
            "no trainable rows — document has no non-empty PROSE/INSTRUCTION/PREFERENCE sections"
        )

    if weights:
        rows = expand_rows_by_weight(rows, weights, seed=seed)
        if not rows:
            raise ValueError(
                "weights dropped every row — check `training.yaml` weights for zeros across all tag values"
            )

    return split(rows, val_frac=val_frac, seed=seed)


def _is_preference_row(row: Row) -> bool:
    return (
        row.get("prompt") is not None
        and row.get("chosen") is not None
        and row.get("rejected") is not None
    )
