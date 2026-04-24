"""Unit tests for the in-memory injected-probe queue."""

from __future__ import annotations

import threading

import pytest

from dlm.train.inject import InjectedProbe, InjectedProbeQueue, QueueFullError


def _probe(prompt: str = "q", reference: str = "a") -> InjectedProbe:
    return InjectedProbe(prompt=prompt, reference=reference)


class TestQueue:
    def test_enqueue_and_drain_fifo(self) -> None:
        q = InjectedProbeQueue(capacity=4)
        for i in range(3):
            q.enqueue(_probe(prompt=f"q{i}"))
        drained = q.drain()
        assert [p.prompt for p in drained] == ["q0", "q1", "q2"]
        assert q.depth() == 0

    def test_drain_empty_returns_empty(self) -> None:
        q = InjectedProbeQueue(capacity=4)
        assert q.drain() == []

    def test_capacity_enforced(self) -> None:
        q = InjectedProbeQueue(capacity=2)
        q.enqueue(_probe())
        q.enqueue(_probe())
        with pytest.raises(QueueFullError):
            q.enqueue(_probe())

    def test_capacity_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="capacity must be positive"):
            InjectedProbeQueue(capacity=0)

    def test_capacity_property_reflects_configured_limit(self) -> None:
        q = InjectedProbeQueue(capacity=8)
        assert q.capacity == 8

    def test_depth_reports_current(self) -> None:
        q = InjectedProbeQueue(capacity=8)
        assert q.depth() == 0
        q.enqueue(_probe())
        q.enqueue(_probe())
        assert q.depth() == 2

    def test_thread_safe_enqueue(self) -> None:
        # Smoke test: two threads enqueue concurrently; no drops, no
        # interleaved corruption.
        q = InjectedProbeQueue(capacity=200)

        def push(n: int) -> None:
            for i in range(n):
                q.enqueue(_probe(prompt=f"t{i}"))

        t1 = threading.Thread(target=push, args=(50,))
        t2 = threading.Thread(target=push, args=(50,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert q.depth() == 100

    def test_drain_concurrent_with_enqueue(self) -> None:
        q = InjectedProbeQueue(capacity=200)
        for i in range(10):
            q.enqueue(_probe(prompt=f"first-{i}"))
        drained = q.drain()
        assert len(drained) == 10
        q.enqueue(_probe(prompt="after"))
        drained2 = q.drain()
        assert len(drained2) == 1
        assert drained2[0].prompt == "after"

    def test_probe_records_accepted_at(self) -> None:
        probe = _probe()
        # Default factory populates it at construction.
        assert probe.accepted_at.tzinfo is not None

    def test_probe_tags_tuple(self) -> None:
        probe = InjectedProbe(prompt="q", reference="a", tags=("sway-ci", "high-priority"))
        assert probe.tags == ("sway-ci", "high-priority")
