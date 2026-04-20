"""Pure-logic debouncer for watch-mode events.

Editors emit 1–3 filesystem events per save (vim writes a swap file,
renames it, truncates the original; VS Code writes the file
in-place but fsyncs separately). The debouncer collapses a burst
into a single `fire` opportunity once the filesystem has been quiet
for a configurable interval.

Time source is injected (`time_fn`) so tests drive the clock with
`time.perf_counter` substitutes instead of sleeping. Thread-safety
is NOT built in — the watch loop runs on a single thread; if a
future sprint splits receive + handle onto separate threads, this
class grows a lock.
"""

from __future__ import annotations

from collections.abc import Callable


class Debouncer:
    """Fires once after `quiet_seconds` of inactivity since the last event.

    Usage pattern (polled from the watch loop):

        deb = Debouncer(quiet_seconds=0.4)
        for event in stream:
            deb.record(time_now)
            ...
        if deb.should_fire(time_now):
            deb.reset()
            handle_change()

    Explicit `record` + `should_fire` + `reset` keeps the loop
    readable; callers that want timer-callback semantics can wrap
    this in a thread.
    """

    def __init__(
        self,
        quiet_seconds: float,
        *,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        if quiet_seconds <= 0:
            raise ValueError(f"quiet_seconds must be > 0, got {quiet_seconds}")
        self._quiet = quiet_seconds
        self._time_fn = time_fn
        self._last_event_at: float | None = None
        self._event_count: int = 0

    @property
    def pending_count(self) -> int:
        """Number of events accumulated since last `reset`.

        Useful in log messages: "coalesced 5 saves into 1 retrain".
        """
        return self._event_count

    def _now(self) -> float:
        if self._time_fn is not None:
            return self._time_fn()
        import time

        return time.monotonic()

    def record(self, at_time: float | None = None) -> None:
        """Register a new event. Resets the quiet timer."""
        now = at_time if at_time is not None else self._now()
        self._last_event_at = now
        self._event_count += 1

    def should_fire(self, at_time: float | None = None) -> bool:
        """True iff an event was recorded AND quiet interval has elapsed."""
        if self._last_event_at is None:
            return False
        now = at_time if at_time is not None else self._now()
        return (now - self._last_event_at) >= self._quiet

    def reset(self) -> None:
        """Clear the pending-event state after firing."""
        self._last_event_at = None
        self._event_count = 0
