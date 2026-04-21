"""In-memory probe queue — the push path for sway-driven training.

Probes POSTed to the RPC endpoint land here until the next cycle
boundary, at which point `drain()` hands them to `build_dataset`. The
queue is bounded (default 1000); enqueue past capacity raises
`QueueFullError` so the RPC layer can map that to HTTP 429.

The dataclass is frozen so a drained probe can be attached verbatim to
the training summary's audit trail without defensive copies.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime


class QueueFullError(Exception):
    """Raised when `InjectedProbeQueue.enqueue` is called past capacity."""


@dataclass(frozen=True)
class InjectedProbe:
    """One probe pushed over the RPC channel.

    `accepted_at` is set by the queue when the probe is enqueued; it's
    the server-side receipt timestamp, not whatever the client claimed.
    """

    prompt: str
    reference: str
    tags: tuple[str, ...] = ()
    source_addr: str = ""
    accepted_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class InjectedProbeQueue:
    """Thread-safe FIFO for RPC-pushed probes.

    The RPC server writes; the cycle-boundary hook drains. One lock
    guards both sides; contention is negligible at the expected rate
    (probes per minute, not per ms).
    """

    def __init__(self, *, capacity: int = 1000) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._items: deque[InjectedProbe] = deque()
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def enqueue(self, probe: InjectedProbe) -> None:
        """Append `probe` to the queue, or raise `QueueFullError`."""
        with self._lock:
            if len(self._items) >= self._capacity:
                raise QueueFullError(
                    f"probe queue at capacity ({self._capacity}); "
                    "drain the queue or raise --queue-capacity"
                )
            self._items.append(probe)

    def drain(self) -> list[InjectedProbe]:
        """Remove and return all queued probes in FIFO order.

        Called at the top of each training cycle's `build_dataset`.
        After this call the queue is empty until the next RPC push.
        """
        with self._lock:
            drained = list(self._items)
            self._items.clear()
            return drained

    def depth(self) -> int:
        with self._lock:
            return len(self._items)
