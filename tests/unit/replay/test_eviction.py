"""Eviction — oldest-first, protected-ids-never."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from dlm.replay.eviction import evict_until
from dlm.replay.models import IndexEntry

_BASE = datetime(2026, 4, 1)


def _entries(n: int, *, length: int = 500) -> list[IndexEntry]:
    return [
        IndexEntry(
            section_id=f"{i:016x}",
            byte_offset=i * length,
            length=length,
            added_at=_BASE + timedelta(days=i),
        )
        for i in range(n)
    ]


class TestEvictionBasics:
    def test_noop_when_under_cap(self) -> None:
        entries = _entries(5)  # 2500 bytes
        kept, evicted = evict_until(entries, max_bytes=100_000)
        assert len(kept) == 5
        assert evicted == []

    def test_evicts_oldest_first(self) -> None:
        entries = _entries(10)  # 5000 bytes total
        kept, evicted = evict_until(entries, max_bytes=2000)
        # Oldest-first: index 0, 1, 2, ... evicted.
        assert evicted == [f"{i:016x}" for i in range(6)]
        assert {e.section_id for e in kept} == {f"{i:016x}" for i in range(6, 10)}

    def test_preserves_order_in_kept(self) -> None:
        """`kept_entries` is in the original input order, not reshuffled."""
        entries = _entries(10)
        kept, _ = evict_until(entries, max_bytes=2000)
        kept_ids = [e.section_id for e in kept]
        assert kept_ids == sorted(kept_ids)  # input was ordered by index

    def test_negative_max_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_bytes"):
            evict_until(_entries(3), max_bytes=-1)


class TestProtection:
    def test_protected_never_evicted_even_if_oldest(self) -> None:
        entries = _entries(10)
        # Protect the two oldest.
        protect = {f"{0:016x}", f"{1:016x}"}
        kept, evicted = evict_until(entries, max_bytes=2000, protect_ids=protect)
        kept_ids = {e.section_id for e in kept}
        assert protect.issubset(kept_ids)
        assert not protect.intersection(evicted)

    def test_hard_reset_preserves_protected(self) -> None:
        entries = _entries(10)
        protect = {f"{5:016x}"}
        kept, _ = evict_until(entries, max_bytes=0, protect_ids=protect)
        assert [e.section_id for e in kept] == [f"{5:016x}"]

    def test_all_protected_keeps_everything(self) -> None:
        entries = _entries(3)
        protect = {e.section_id for e in entries}
        kept, evicted = evict_until(entries, max_bytes=0, protect_ids=protect)
        assert len(kept) == 3
        assert evicted == []


class TestDeterminism:
    def test_tie_broken_by_section_id(self) -> None:
        """Two entries with identical added_at break ties on section_id."""
        same = _BASE
        entries = [
            IndexEntry(section_id="b" * 16, byte_offset=0, length=1000, added_at=same),
            IndexEntry(section_id="a" * 16, byte_offset=1000, length=1000, added_at=same),
        ]
        # Cap forces ONE eviction; the ties-sort must pick "a"*16 first
        # since it's lex-lower.
        _, evicted = evict_until(entries, max_bytes=1000)
        assert evicted == ["a" * 16]
