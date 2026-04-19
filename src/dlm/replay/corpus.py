"""Append-only zstd-framed corpus writer + random-access reader.

Each frame is **independently decodable**: a self-contained zstd level-3
frame wrapping a CBOR-encoded `SectionSnapshot`. Frame independence is
what makes random-access reads possible — given an `IndexEntry` with
`byte_offset` and `length`, we can decompress one frame without touching
the rest of `corpus.zst`.

Design notes
------------

- **Compression level fixed at zstd 3.** Determinism contract: the same
  bytes in → the same bytes out. Changing level changes the output and
  breaks the corpus-reproducibility CI check.
- **CBOR over JSON** for the record body: ~30% smaller for typical
  text + the encoder is deterministic (canonical key order).
- **`O_APPEND` semantics** for writes: we open + append + close for
  each snapshot. Concurrent writers are caller's problem (store lock
  in Sprint 04 handles it); this module assumes single-writer access.
- **No partial-frame recovery.** If a write crashes mid-frame, the
  corpus tail may have a garbage zstd frame. The caller is expected
  to run `verify_tail()` after lock acquisition; if it reports
  corruption, the caller decides whether to truncate or rebuild.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

import cbor2
import zstandard as zstd

from dlm.replay.errors import CorpusCorruptError
from dlm.replay.models import IndexEntry, SectionSnapshot

if TYPE_CHECKING:
    from collections.abc import Iterator

_ZSTD_LEVEL = 3


def append_snapshot(corpus_path: Path, snapshot: SectionSnapshot) -> IndexEntry:
    """Append one snapshot to `corpus.zst`, return its `IndexEntry`.

    The entry's `added_at` is the snapshot's `last_seen_at` — the
    sampler's recency weight uses this as the source of truth, not
    wall-clock at read time.
    """
    frame = _encode_frame(snapshot)
    offset = _file_size(corpus_path)
    with corpus_path.open("ab") as fh:
        fh.write(frame)
    return IndexEntry(
        section_id=snapshot.section_id,
        byte_offset=offset,
        length=len(frame),
        added_at=snapshot.last_seen_at,
    )


def read_chunk(corpus_path: Path, *, byte_offset: int, length: int) -> SectionSnapshot:
    """Decompress + decode one frame at `byte_offset`, `length` bytes long."""
    with corpus_path.open("rb") as fh:
        fh.seek(byte_offset)
        frame = fh.read(length)
    if len(frame) != length:
        raise CorpusCorruptError(
            f"short read: expected {length} bytes, got {len(frame)}",
            byte_offset=byte_offset,
            length=length,
        )
    return _decode_frame(frame, byte_offset=byte_offset, length=length)


def iter_snapshots(corpus_path: Path, entries: list[IndexEntry]) -> Iterator[SectionSnapshot]:
    """Yield snapshots for each entry in order.

    Opens the corpus once and reuses the handle across reads so a
    full sweep doesn't re-open the file per entry.
    """
    with corpus_path.open("rb") as fh:
        for entry in entries:
            fh.seek(entry.byte_offset)
            frame = fh.read(entry.length)
            if len(frame) != entry.length:
                raise CorpusCorruptError(
                    f"short read: expected {entry.length} bytes, got {len(frame)}",
                    byte_offset=entry.byte_offset,
                    length=entry.length,
                )
            yield _decode_frame(frame, byte_offset=entry.byte_offset, length=entry.length)


# --- internals ---------------------------------------------------------------


def _encode_frame(snapshot: SectionSnapshot) -> bytes:
    # `timezone=UTC` lets cbor2 encode tz-naive datetimes (our manifest
    # convention) as UTC instants; `canonical=True` picks deterministic
    # map key ordering so frame bytes are reproducible across runs.
    payload = cbor2.dumps(
        snapshot.model_dump(mode="python"),
        timezone=UTC,
        canonical=True,
    )
    compressor = zstd.ZstdCompressor(level=_ZSTD_LEVEL)
    return compressor.compress(payload)


def _decode_frame(frame: bytes, *, byte_offset: int, length: int) -> SectionSnapshot:
    try:
        decompressor = zstd.ZstdDecompressor()
        payload = decompressor.decompress(frame)
    except zstd.ZstdError as exc:
        raise CorpusCorruptError(
            f"zstd decompression failed: {exc}",
            byte_offset=byte_offset,
            length=length,
        ) from exc
    try:
        record = cbor2.loads(payload)
    except (cbor2.CBORDecodeError, ValueError) as exc:
        raise CorpusCorruptError(
            f"CBOR decode failed: {exc}",
            byte_offset=byte_offset,
            length=length,
        ) from exc
    try:
        return SectionSnapshot.model_validate(record)
    except Exception as exc:
        raise CorpusCorruptError(
            f"snapshot validation failed: {exc}",
            byte_offset=byte_offset,
            length=length,
        ) from exc


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0
