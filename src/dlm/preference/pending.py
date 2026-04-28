"""Persist staged auto-mined preference sections between CLI steps.

`dlm preference mine` is dry-run against the source `.dlm` by default,
but the follow-up `dlm preference apply` still needs a stable plan to
write. This module stores the mined `Section` payloads under the store
root so the reviewed plan can be applied later without re-running
sampling or judging.

I/O is shared with `dlm.synth.pending` via `dlm._pending`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dlm._pending import (
    PendingSectionPlan,
    _optional_float,
    _optional_int,
    _optional_str,
    _section_from_payload,
    _section_to_payload,
)
from dlm._pending import (
    clear_pending_plan as _clear,
)
from dlm._pending import (
    load_pending_plan as _load,
)
from dlm._pending import (
    pending_plan_path as _path,
)
from dlm._pending import (
    save_pending_plan as _save,
)
from dlm.preference.errors import PreferenceMiningError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from dlm.doc.sections import Section
    from dlm.store.paths import StorePath


_SUBDIR = "preference"
_LABEL = "preference plan"


class PendingPreferencePlanError(PreferenceMiningError):
    """Raised when the staged preference plan cannot be read or validated."""


@dataclass(frozen=True)
class PendingPreferencePlan(PendingSectionPlan):
    """One staged preference-mine plan for a store."""


def pending_plan_path(store: StorePath) -> Path:
    """Path to the staged preference-mine payload for `store`."""
    return _path(store, subdir=_SUBDIR)


def save_pending_plan(
    store: StorePath,
    *,
    source_path: Path,
    sections: Sequence[Section],
) -> PendingPreferencePlan:
    """Persist `sections` as the staged plan for `store`."""
    return _save(  # type: ignore[return-value]
        store,
        source_path=source_path,
        sections=sections,
        subdir=_SUBDIR,
        plan_cls=PendingPreferencePlan,
    )


def load_pending_plan(store: StorePath) -> PendingPreferencePlan | None:
    """Return the staged plan for `store`, or None when absent."""
    return _load(  # type: ignore[return-value]
        store,
        subdir=_SUBDIR,
        plan_cls=PendingPreferencePlan,
        error_cls=PendingPreferencePlanError,
        label=_LABEL,
    )


def clear_pending_plan(store: StorePath) -> bool:
    """Delete the staged plan for `store`. Returns True iff it existed."""
    return _clear(store, subdir=_SUBDIR)


# Re-export private I/O helpers so test modules that reached into the
# original implementation continue to work without import churn.
__all__ = [
    "PendingPreferencePlan",
    "PendingPreferencePlanError",
    "_optional_float",
    "_optional_int",
    "_optional_str",
    "_section_from_payload",
    "_section_to_payload",
    "clear_pending_plan",
    "load_pending_plan",
    "pending_plan_path",
    "save_pending_plan",
]
