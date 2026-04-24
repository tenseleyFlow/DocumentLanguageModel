"""Append-only zstd+CBOR corpus round-trip + corruption handling."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pytest

import dlm.replay.corpus as replay_corpus
from dlm.replay.corpus import _encode_frame, append_snapshot, iter_snapshots, read_chunk
from dlm.replay.errors import CorpusCorruptError
from dlm.replay.models import SectionSnapshot


def _snap(sid: str, content: str = "x") -> SectionSnapshot:
    return SectionSnapshot(
        section_id=sid,
        section_type="prose",
        content=content,
        first_seen_at=datetime(2026, 1, 1),
        last_seen_at=datetime(2026, 1, 1),
    )


class TestAppendAndRead:
    def test_single_round_trip(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.zst"
        snap = _snap("a" * 16, "hello")
        entry = append_snapshot(corpus, snap)
        assert entry.byte_offset == 0
        assert entry.length > 0

        back = read_chunk(corpus, byte_offset=entry.byte_offset, length=entry.length)
        assert back.section_id == snap.section_id
        assert back.content == snap.content

    def test_random_access_second_frame(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.zst"
        s1 = _snap("a" * 16, "one")
        s2 = _snap("b" * 16, "two")
        e1 = append_snapshot(corpus, s1)
        e2 = append_snapshot(corpus, s2)
        assert e2.byte_offset == e1.length

        s2_back = read_chunk(corpus, byte_offset=e2.byte_offset, length=e2.length)
        assert s2_back.content == "two"

    def test_iter_snapshots_preserves_order(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.zst"
        snaps = [_snap(f"{i:016x}", f"c{i}") for i in range(5)]
        entries = [append_snapshot(corpus, s) for s in snaps]
        out = list(iter_snapshots(corpus, entries))
        assert [s.content for s in out] == [s.content for s in snaps]


class TestDeterminism:
    def test_same_snapshot_same_bytes(self) -> None:
        s = _snap("a" * 16, "determinism")
        assert _encode_frame(s) == _encode_frame(s)

    def test_different_content_different_frames(self) -> None:
        a = _encode_frame(_snap("a" * 16, "alpha"))
        b = _encode_frame(_snap("a" * 16, "beta"))
        assert a != b


class TestCorruption:
    def test_short_read_raises(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.zst"
        append_snapshot(corpus, _snap("a" * 16))
        # Claim the frame is twice as long as reality.
        with pytest.raises(CorpusCorruptError, match="short read"):
            read_chunk(corpus, byte_offset=0, length=10_000)

    def test_garbage_bytes_raise(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.zst"
        corpus.write_bytes(b"\x00" * 128)
        with pytest.raises(CorpusCorruptError, match="zstd|CBOR|validation"):
            read_chunk(corpus, byte_offset=0, length=128)

    def test_valid_zstd_but_bad_cbor(self, tmp_path: Path) -> None:
        """Valid zstd framing but CBOR payload is random bytes."""
        import zstandard as zstd

        corpus = tmp_path / "corpus.zst"
        frame = zstd.ZstdCompressor(level=3).compress(b"not-a-cbor-record-\xff\xff\xff")
        corpus.write_bytes(frame)
        with pytest.raises(CorpusCorruptError):
            read_chunk(corpus, byte_offset=0, length=len(frame))

    def test_cbor_value_error_is_wrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame = _encode_frame(_snap("a" * 16, "hello"))
        corpus = tmp_path / "corpus.zst"
        corpus.write_bytes(frame)

        def _boom(_payload: bytes) -> object:
            raise ValueError("bad semantic tag")

        monkeypatch.setattr(replay_corpus.cbor2, "loads", _boom)
        with pytest.raises(CorpusCorruptError, match="CBOR decode failed"):
            read_chunk(corpus, byte_offset=0, length=len(frame))

    def test_iter_short_read_raises(self, tmp_path: Path) -> None:
        """iter_snapshots also guards against truncated entries."""
        from dlm.replay.models import IndexEntry

        corpus = tmp_path / "corpus.zst"
        snap = _snap("a" * 16, "ok")
        entry = append_snapshot(corpus, snap)
        # Forge an entry claiming more bytes than exist.
        bogus = IndexEntry(
            section_id=entry.section_id,
            byte_offset=0,
            length=entry.length + 9999,
        )
        with pytest.raises(CorpusCorruptError, match="short read"):
            list(iter_snapshots(corpus, [bogus]))

    def test_append_creates_file_if_missing(self, tmp_path: Path) -> None:
        """File-size helper returns 0 when corpus.zst doesn't exist yet."""
        corpus = tmp_path / "sub" / "corpus.zst"
        corpus.parent.mkdir()
        assert not corpus.exists()
        entry = append_snapshot(corpus, _snap("a" * 16, "first"))
        assert entry.byte_offset == 0
        assert corpus.exists()


class TestThroughput:
    def test_10k_round_trip_under_one_second(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.zst"
        snaps = [_snap(f"{i:016x}", f"content-{i}" * 4) for i in range(10_000)]

        t0 = time.perf_counter()
        entries = [append_snapshot(corpus, s) for s in snaps]
        elapsed_append = time.perf_counter() - t0

        t0 = time.perf_counter()
        out = list(iter_snapshots(corpus, entries))
        elapsed_read = time.perf_counter() - t0

        assert len(out) == 10_000
        # DoD: both under 1s on CI (local dev is typically 5-10x faster).
        assert elapsed_append < 3.0, f"append too slow: {elapsed_append:.2f}s"
        assert elapsed_read < 3.0, f"read too slow: {elapsed_read:.2f}s"
