"""WatchStatus + render_status format stability."""

from __future__ import annotations

from dlm.watch.status import WatchState, WatchStatus, render_status


class TestRenderStatus:
    def test_initial_state_shows_idle(self) -> None:
        status = WatchStatus(doc_path="mydoc.dlm", sections=12)
        line = render_status(status, now=0.0)
        assert "Watching mydoc.dlm" in line
        assert "state: idle" in line
        assert "sections: 12" in line
        assert "last train: never" in line

    def test_after_cycle_shows_loss_and_steps(self) -> None:
        status = WatchStatus(doc_path="mydoc.dlm", sections=12)
        status.mark_cycle_done(
            train_loss=1.2, val_loss=1.35, steps=50, coalesced=1
        )
        line = render_status(status)
        assert "val loss: 1.35" in line
        assert "steps: 50" in line

    def test_coalesced_only_shown_when_plural(self) -> None:
        status = WatchStatus(doc_path="d")
        status.mark_cycle_done(
            train_loss=None, val_loss=None, steps=10, coalesced=1
        )
        assert "coalesced" not in render_status(status)

        status.mark_cycle_done(
            train_loss=None, val_loss=None, steps=10, coalesced=5
        )
        assert "coalesced: 5" in render_status(status)

    def test_age_buckets(self) -> None:
        status = WatchStatus(doc_path="d")
        status.last_cycle_at = 0.0
        assert "12s ago" in render_status(status, now=12.0)
        assert "2m ago" in render_status(status, now=125.0)
        assert "1h ago" in render_status(status, now=3700.0)


class TestWatchState:
    def test_enum_values(self) -> None:
        assert WatchState.IDLE.value == "idle"
        assert WatchState.TRAINING.value == "training"
        assert WatchState.CANCELLED.value == "cancelled"
