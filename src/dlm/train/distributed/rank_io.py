"""Per-rank I/O helpers for DDP training.

Every rank runs the same training loop, but only rank 0 writes
checkpoints, logs, and manifests — otherwise we'd get N copies of the
same files (best case) or corrupted interleaved writes (worst case).

The helpers are lightweight wrappers around `accelerate.Accelerator`
state. We read the `is_main_process` / `wait_for_everyone` surface off
a passed-in accelerator so tests can substitute a mock.

Masking policy:
- `master_only(fn)` — decorator. Returns `fn(*args, **kwargs)` on rank
  0, None on everyone else.
- `barrier(accelerator)` — wait for all ranks; thin wrapper over
  `accelerator.wait_for_everyone`.
- `gather_metrics(accelerator, values)` — all-reduce numeric metrics
  across ranks, returning the mean. Used for the loss value reported
  by the logger.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_T = TypeVar("_T")


def is_main_process(accelerator: Any) -> bool:
    """True on rank 0; True in single-process mode too.

    Defaults `getattr(accelerator, "is_main_process", True)` so a
    single-process caller that passes `None` still does its I/O.
    """
    if accelerator is None:
        return True
    return bool(getattr(accelerator, "is_main_process", True))


def master_only(fn: Callable[..., _T]) -> Callable[..., _T | None]:
    """Decorator: only rank 0 runs `fn`; others get `None`.

    Expects the first positional argument to be the accelerator (or
    None in single-process). Mirrors the nanoGPT pattern called out in
    findings §6.
    """

    @wraps(fn)
    def wrapper(accelerator: Any, *args: Any, **kwargs: Any) -> _T | None:
        if not is_main_process(accelerator):
            return None
        return fn(accelerator, *args, **kwargs)

    return wrapper


def barrier(accelerator: Any) -> None:
    """Block until every rank reaches this point.

    No-op on single-process (`accelerator is None` or no
    `wait_for_everyone`). Typical use: after rank 0 writes a
    checkpoint, call `barrier` so other ranks don't racy-read the
    partially-written files.
    """
    if accelerator is None:
        return
    waiter = getattr(accelerator, "wait_for_everyone", None)
    if callable(waiter):
        waiter()


def gather_metrics(
    accelerator: Any, values: Mapping[str, float] | Sequence[tuple[str, float]]
) -> dict[str, float]:
    """Cross-rank mean of a dict of scalar metrics.

    On single-process (`accelerator is None`) returns `dict(values)`
    unchanged. Multi-rank path uses `accelerator.gather_for_metrics`
    which is the documented Accelerate surface for this.

    Keeps the impl defensive: if the accelerator doesn't expose
    `gather_for_metrics` (older versions or a stub), returns the
    unchanged mapping rather than raising — the alternative is to
    break the training loop for a reporting-only helper.
    """
    as_dict = dict(values)
    if accelerator is None:
        return as_dict
    gather = getattr(accelerator, "gather_for_metrics", None)
    if not callable(gather):
        return as_dict

    # We call gather per-metric to avoid building a tensor for a
    # heterogeneous dict. For a shape-stable dict of floats this is
    # clearer than stacking. torch is a core runtime dependency, so
    # the import is always available.
    import torch

    reduced: dict[str, float] = {}
    for name, value in as_dict.items():
        tensor = torch.tensor([float(value)])
        gathered = gather(tensor)
        if gathered is None:
            reduced[name] = float(value)
            continue
        reduced[name] = float(gathered.mean().item())
    return reduced
