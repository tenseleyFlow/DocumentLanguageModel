"""Persist staged auto-synth instruction sections between CLI steps.

I/O is shared with `dlm.preference.pending` via `dlm._pending`.
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
from dlm.synth.errors import SynthError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from dlm.doc.sections import Section
    from dlm.store.paths import StorePath


_SUBDIR = "synth"
_LABEL = "synth plan"


class PendingSynthPlanError(SynthError):
    """Raised when the staged synth plan cannot be read or validated."""


@dataclass(frozen=True)
class PendingSynthPlan(PendingSectionPlan):
    """One staged synth plan for a store."""


def pending_plan_path(store: StorePath) -> Path:
    """Path to the staged synth payload for `store`."""
    return _path(store, subdir=_SUBDIR)


def save_pending_plan(
    store: StorePath,
    *,
    source_path: Path,
    sections: Sequence[Section],
) -> PendingSynthPlan:
    """Persist `sections` as the staged synth plan for `store`."""
    return _save(  # type: ignore[return-value]
        store,
        source_path=source_path,
        sections=sections,
        subdir=_SUBDIR,
        plan_cls=PendingSynthPlan,
    )


def load_pending_plan(store: StorePath) -> PendingSynthPlan | None:
    """Return the staged synth plan for `store`, or None when absent."""
    return _load(  # type: ignore[return-value]
        store,
        subdir=_SUBDIR,
        plan_cls=PendingSynthPlan,
        error_cls=PendingSynthPlanError,
        label=_LABEL,
    )


def clear_pending_plan(store: StorePath) -> bool:
    """Delete the staged synth plan for `store`. Returns True iff it existed."""
    return _clear(store, subdir=_SUBDIR)


__all__ = [
    "PendingSynthPlan",
    "PendingSynthPlanError",
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
