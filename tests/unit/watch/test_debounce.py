"""Debouncer — pure time-source injection."""

from __future__ import annotations

import pytest

from dlm.watch.debounce import Debouncer


class _Clock:
    """Injectable monotonic clock for deterministic tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


class TestDebouncerFires:
    def test_idle_does_not_fire(self) -> None:
        clock = _Clock()
        d = Debouncer(quiet_seconds=0.4, time_fn=clock)
        assert d.should_fire() is False

    def test_fires_after_quiet_interval(self) -> None:
        clock = _Clock()
        d = Debouncer(quiet_seconds=0.4, time_fn=clock)
        d.record()
        clock.advance(0.5)
        assert d.should_fire() is True

    def test_does_not_fire_during_quiet_window(self) -> None:
        clock = _Clock()
        d = Debouncer(quiet_seconds=0.4, time_fn=clock)
        d.record()
        clock.advance(0.2)
        assert d.should_fire() is False

    def test_new_event_resets_quiet_window(self) -> None:
        clock = _Clock()
        d = Debouncer(quiet_seconds=0.4, time_fn=clock)
        d.record()
        clock.advance(0.3)
        d.record()  # another event; reset
        clock.advance(0.3)  # only 0.3s since last event
        assert d.should_fire() is False
        clock.advance(0.2)
        assert d.should_fire() is True


class TestDebouncerCoalesces:
    def test_counts_events(self) -> None:
        d = Debouncer(quiet_seconds=0.4, time_fn=_Clock())
        assert d.pending_count == 0
        d.record()
        d.record()
        d.record()
        assert d.pending_count == 3

    def test_reset_clears_state(self) -> None:
        clock = _Clock()
        d = Debouncer(quiet_seconds=0.4, time_fn=clock)
        d.record()
        d.record()
        d.reset()
        assert d.pending_count == 0
        assert d.should_fire() is False


class TestDebouncerValidation:
    def test_non_positive_quiet_rejected(self) -> None:
        with pytest.raises(ValueError, match="quiet_seconds"):
            Debouncer(quiet_seconds=0)
        with pytest.raises(ValueError, match="quiet_seconds"):
            Debouncer(quiet_seconds=-0.1)
