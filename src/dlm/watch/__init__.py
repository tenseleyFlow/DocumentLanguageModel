"""`dlm train --watch` — save-to-train loop.

Heavy imports (`watchfiles`, the trainer) are deferred so this
module's collection cost is small. Public surface:

- `Debouncer` (pure logic)
- `run_watch` (the loop)
- `WatchStatus` (observable state for the status line)
"""

from __future__ import annotations

from dlm.watch.debounce import Debouncer
from dlm.watch.errors import WatchError, WatchSetupError
from dlm.watch.status import WatchState, WatchStatus, render_status

__all__ = [
    "Debouncer",
    "WatchError",
    "WatchSetupError",
    "WatchState",
    "WatchStatus",
    "render_status",
]
