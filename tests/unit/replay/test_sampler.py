"""Sampler determinism, recency bias, and contract."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest

from dlm.replay.errors import SamplerError
from dlm.replay.models import IndexEntry
from dlm.replay.sampler import _weighted_reservoir, sample

_NOW = datetime(2026, 4, 18)


def _entries(n: int, *, age_step_days: float = 1.0) -> list[IndexEntry]:
    return [
        IndexEntry(
            section_id=f"{i:016x}",
            byte_offset=i * 100,
            length=100,
            added_at=_NOW - timedelta(days=i * age_step_days),
        )
        for i in range(n)
    ]


class TestDeterminism:
    def test_same_seed_same_sample(self) -> None:
        entries = _entries(100)
        p1 = sample(entries, k=10, now=_NOW, rng=random.Random(42))
        p2 = sample(entries, k=10, now=_NOW, rng=random.Random(42))
        assert [e.section_id for e in p1] == [e.section_id for e in p2]

    def test_different_seed_different_sample(self) -> None:
        entries = _entries(100)
        p1 = sample(entries, k=10, now=_NOW, rng=random.Random(0))
        p2 = sample(entries, k=10, now=_NOW, rng=random.Random(1))
        assert [e.section_id for e in p1] != [e.section_id for e in p2]


class TestBoundaries:
    def test_empty_entries(self) -> None:
        assert sample([], k=5, now=_NOW, rng=random.Random(0)) == []

    def test_k_zero_returns_empty(self) -> None:
        assert sample(_entries(10), k=0, now=_NOW, rng=random.Random(0)) == []

    def test_k_negative_rejected(self) -> None:
        with pytest.raises(SamplerError, match="non-negative"):
            sample(_entries(10), k=-1, now=_NOW, rng=random.Random(0))

    def test_k_exceeds_size_returns_all(self) -> None:
        entries = _entries(5)
        out = sample(entries, k=10, now=_NOW, rng=random.Random(0))
        assert len(out) == 5
        assert [e.section_id for e in out] == sorted(e.section_id for e in entries)

    def test_unknown_scheme_raises(self) -> None:
        with pytest.raises(SamplerError, match="scheme"):
            sample(
                _entries(5),
                k=3,
                now=_NOW,
                rng=random.Random(0),
                scheme="frequency",  # type: ignore[arg-type]
            )


class TestRecencyBias:
    def test_recent_sections_more_likely(self) -> None:
        """Average age of recency picks must be younger than uniform over many seeds."""
        entries = _entries(200, age_step_days=2.0)
        recency_total_age = 0
        uniform_total_age = 0
        trials = 30
        for seed in range(trials):
            rec = sample(entries, k=20, now=_NOW, rng=random.Random(seed), scheme="recency")
            uni = sample(entries, k=20, now=_NOW, rng=random.Random(seed), scheme="uniform")
            recency_total_age += sum((_NOW - e.added_at).days for e in rec)
            uniform_total_age += sum((_NOW - e.added_at).days for e in uni)
        assert recency_total_age < uniform_total_age

    def test_zero_weight_entry_never_sampled(self) -> None:
        entries = [
            IndexEntry(
                section_id="a" * 16,
                byte_offset=0,
                length=100,
                added_at=_NOW,
                weight=1.0,
            ),
            IndexEntry(
                section_id="b" * 16,
                byte_offset=100,
                length=100,
                added_at=_NOW,
                weight=0.0,
            ),
        ]
        # k=1, 20 trials; weight=0 must never appear.
        for seed in range(20):
            picked = sample(entries, k=1, now=_NOW, rng=random.Random(seed))
            assert picked[0].section_id == "a" * 16


class TestStableOrdering:
    def test_stable_input_ordering_irrespective_of_list_order(self) -> None:
        """Sampling depends only on RNG + entries content, not on list order."""
        entries_a = _entries(50)
        entries_b = list(reversed(entries_a))
        p1 = sample(entries_a, k=5, now=_NOW, rng=random.Random(7), scheme="uniform")
        p2 = sample(entries_b, k=5, now=_NOW, rng=random.Random(7), scheme="uniform")
        assert [e.section_id for e in p1] == [e.section_id for e in p2]


class TestReservoirEdgeCases:
    def test_zero_random_draw_retries_and_falls_back_to_tiny_positive(self) -> None:
        entries = _entries(2)

        class _ZeroThenHalfRng:
            def __init__(self) -> None:
                self._values = iter([0.0, 0.0, 0.5, 0.5])

            def random(self) -> float:
                return next(self._values)

        picked = sample(entries, k=1, now=_NOW, rng=_ZeroThenHalfRng(), scheme="uniform")
        assert len(picked) == 1

    def test_nonpositive_weight_entries_are_skipped(self) -> None:
        entries = _entries(3)
        picked = _weighted_reservoir(
            entries,
            weights=[1.0, 0.0, -1.0],
            k=3,
            rng=random.Random(0),
        )
        assert [entry.section_id for entry in picked] == [entries[0].section_id]
