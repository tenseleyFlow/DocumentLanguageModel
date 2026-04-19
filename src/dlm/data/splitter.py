"""Deterministic train / val split keyed on (seed, section_id).

The invariant: adding a section to the `.dlm` does NOT reshuffle the
existing assignments. Every row's train-vs-val fate is a pure function
of `(seed, row["_dlm_section_id"], sub_index)` — the sub-index is the
row's position within its section (so a single INSTRUCTION block with
ten Q/A pairs distributes those pairs across the split independently).

The split is computed by hashing `(seed, section_id, sub_index)` and
comparing against `val_frac * 2**64`. This is stable across Python
versions (we use `hashlib.sha256` rather than `hash()`).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset

Row = dict[str, Any]


def split(
    rows: list[Row],
    *,
    val_frac: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Partition `rows` into (train_ds, val_ds) datasets.

    `val_frac` must be in (0, 1). `seed` is combined with each row's
    `_dlm_section_id` + its in-section sub-index to produce a stable
    assignment.

    Raises `ValueError` if `val_frac` is outside (0, 1) or if any row
    lacks `_dlm_section_id`.
    """
    from datasets import Dataset

    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac!r}")

    train_rows: list[Row] = []
    val_rows: list[Row] = []
    per_section_index: dict[str, int] = defaultdict(int)

    threshold = int(val_frac * (1 << 64))

    for row in rows:
        sid = row.get("_dlm_section_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError(
                "every row must carry a string `_dlm_section_id` "
                "(did you skip sections_to_rows?)"
            )
        sub_index = per_section_index[sid]
        per_section_index[sid] += 1
        if _assigns_to_val(seed=seed, section_id=sid, sub_index=sub_index, threshold=threshold):
            val_rows.append(row)
        else:
            train_rows.append(row)

    return Dataset.from_list(train_rows), Dataset.from_list(val_rows)


def _assigns_to_val(*, seed: int, section_id: str, sub_index: int, threshold: int) -> bool:
    key = f"{seed}\x00{section_id}\x00{sub_index}".encode()
    digest = hashlib.sha256(key).digest()[:8]
    bucket = int.from_bytes(digest, byteorder="big", signed=False)
    return bucket < threshold
