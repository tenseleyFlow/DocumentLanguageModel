"""Live-status rendering for watch mode.

`WatchStatus` is a mutable snapshot that the watch loop updates
between cycles; `render_status` turns it into a Rich renderable the
CLI can hand to `rich.live.Live` (or plain `console.print` in
non-TTY environments).

Split from the loop so the renderer is unit-testable without running
the loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum


class WatchState(StrEnum):
    """Coarse-grained state reported on the status line."""

    IDLE = "idle"
    """Watcher running; no retrain in flight."""

    TRAINING = "training"
    """Incremental retrain is executing."""

    CANCELLED = "cancelled"
    """User hit Ctrl-C during a cycle; loop is winding down."""


@dataclass
class WatchStatus:
    """Mutable snapshot the watch loop updates each cycle."""

    doc_path: str
    state: WatchState = WatchState.IDLE
    sections: int = 0
    last_cycle_at: float | None = None
    last_val_loss: float | None = None
    last_train_loss: float | None = None
    last_cycle_steps: int | None = None
    coalesced_events: int = 0

    def mark_cycle_done(
        self,
        *,
        train_loss: float | None,
        val_loss: float | None,
        steps: int,
        coalesced: int,
    ) -> None:
        """Called by the loop after a successful retrain."""
        self.state = WatchState.IDLE
        self.last_cycle_at = time.monotonic()
        self.last_train_loss = train_loss
        self.last_val_loss = val_loss
        self.last_cycle_steps = steps
        self.coalesced_events = coalesced


def _format_age(ref_time: float | None, now: float) -> str:
    """Render `last_cycle_at` as a human age ('12s ago', '3m ago')."""
    if ref_time is None:
        return "never"
    delta = max(0.0, now - ref_time)
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    return f"{int(delta // 3600)}h ago"


def render_status(status: WatchStatus, *, now: float | None = None) -> str:
    """Produce the one-line status string.

    Kept as plain text so tests assert exact content. The CLI can
    wrap this in a Rich Panel for color; the data is the same.
    """
    now_s = now if now is not None else time.monotonic()
    bits = [
        f"Watching {status.doc_path}",
        f"state: {status.state.value}",
        f"sections: {status.sections}",
        f"last train: {_format_age(status.last_cycle_at, now_s)}",
    ]
    if status.last_val_loss is not None:
        bits.append(f"val loss: {status.last_val_loss:.2f}")
    if status.last_cycle_steps is not None:
        bits.append(f"steps: {status.last_cycle_steps}")
    if status.coalesced_events > 1:
        bits.append(f"coalesced: {status.coalesced_events}")
    return "  |  ".join(bits)
