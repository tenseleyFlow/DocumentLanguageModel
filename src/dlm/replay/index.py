"""JSON load/save for the corpus index.

The index is a flat array of `IndexEntry` objects. We sort entries by
`section_id` before serializing so byte-identical corpora + identical
insertion orders produce byte-identical index files (CI
reproducibility gate in Sprint 08).

The JSON format is `pydantic.TypeAdapter`-serialized with sorted keys
and a trailing newline. I/O is atomic via `dlm.io.atomic.write_bytes`
so concurrent readers never see a torn file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from dlm.io.atomic import write_bytes
from dlm.replay.errors import IndexCorruptError
from dlm.replay.models import IndexEntry

_INDEX_ADAPTER: TypeAdapter[list[IndexEntry]] = TypeAdapter(list[IndexEntry])


def load_index(path: Path) -> list[IndexEntry]:
    """Return the list of entries at `path`, or `[]` if `path` is missing.

    Raises `IndexCorruptError` if the file exists but isn't a valid
    JSON array of `IndexEntry` records.
    """
    if not path.exists():
        return []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise IndexCorruptError(f"cannot read {path}: {exc}") from exc
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IndexCorruptError(f"{path} is not valid JSON: {exc}") from exc
    try:
        return _INDEX_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise IndexCorruptError(f"{path} has invalid entries: {exc}") from exc


def save_index(path: Path, entries: list[IndexEntry]) -> None:
    """Atomically write `entries` to `path`, sorted by `section_id`.

    Serializes with `mode="json"` so `datetime` fields become ISO-8601
    strings. Parent directory must already exist.
    """
    sorted_entries = sorted(entries, key=lambda e: e.section_id)
    payload = _INDEX_ADAPTER.dump_python(sorted_entries, mode="json")
    blob = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    write_bytes(path, blob)
