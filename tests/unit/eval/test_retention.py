"""Retention slice determinism + stability."""

from __future__ import annotations

from datetime import datetime

import pytest

from dlm.eval.errors import RetentionSliceError
from dlm.eval.retention import build_retention_slice, retention_delta
from dlm.replay.models import IndexEntry


def _entry(sid: str) -> IndexEntry:
    return IndexEntry(
        section_id=sid,
        byte_offset=0,
        length=100,
        added_at=datetime(2026, 1, 1),
    )


class TestBuildRetentionSlice:
    def test_empty_corpus_raises(self) -> None:
        with pytest.raises(RetentionSliceError, match="empty"):
            build_retention_slice([], frac=0.1, seed=0)

    def test_frac_out_of_range_raises(self) -> None:
        entries = [_entry(f"{i:016x}") for i in range(10)]
        with pytest.raises(RetentionSliceError, match="frac"):
            build_retention_slice(entries, frac=0.0, seed=0)
        with pytest.raises(RetentionSliceError, match="frac"):
            build_retention_slice(entries, frac=1.5, seed=0)

    def test_frac_picks_expected_count(self) -> None:
        entries = [_entry(f"{i:016x}") for i in range(100)]
        slice_ = build_retention_slice(entries, frac=0.1, seed=42)
        assert len(slice_.entries) == 10

    def test_small_corpus_picks_at_least_one(self) -> None:
        """5% of 3 is 0.15 → rounds up to 1."""
        entries = [_entry(f"{i:016x}") for i in range(3)]
        slice_ = build_retention_slice(entries, frac=0.05, seed=0)
        assert len(slice_.entries) == 1

    def test_seed_stable(self) -> None:
        entries = [_entry(f"{i:016x}") for i in range(50)]
        a = build_retention_slice(entries, frac=0.1, seed=7)
        b = build_retention_slice(entries, frac=0.1, seed=7)
        assert [e.section_id for e in a.entries] == [e.section_id for e in b.entries]

    def test_different_seeds_different_slice(self) -> None:
        entries = [_entry(f"{i:016x}") for i in range(50)]
        a = build_retention_slice(entries, frac=0.1, seed=1)
        b = build_retention_slice(entries, frac=0.1, seed=999)
        assert {e.section_id for e in a.entries} != {e.section_id for e in b.entries}

    def test_identical_inputs_identical_slice(self) -> None:
        """Same corpus + same seed → identical slice across calls.

        Unlike Sprint 08's splitter, the retention slice is NOT
        growth-stable: adding new entries to the corpus can displace
        existing members from the top-k. The spec only requires
        seed-determinism for a fixed input, which this asserts.
        Cross-run comparability comes from reporting the loss delta on
        whatever slice the current run sees, not from freezing the
        slice membership across corpus growth.
        """
        entries = [_entry(f"{i:016x}") for i in range(100)]
        first = build_retention_slice(entries, frac=0.05, seed=42)
        second = build_retention_slice(entries, frac=0.05, seed=42)
        assert first.section_ids == second.section_ids


class TestRetentionDelta:
    def test_both_present(self) -> None:
        assert retention_delta(
            current_retention_loss=1.5, previous_retention_loss=1.2
        ) == pytest.approx(0.3)

    def test_none_when_either_missing(self) -> None:
        assert retention_delta(current_retention_loss=None, previous_retention_loss=1.0) is None
        assert retention_delta(current_retention_loss=1.0, previous_retention_loss=None) is None
        assert retention_delta(current_retention_loss=None, previous_retention_loss=None) is None
