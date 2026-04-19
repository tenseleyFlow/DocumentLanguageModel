"""CUDA OOM handling — actionable messages + `grad_accum` recommendation.

A CUDA OOM mid-training is one of the most common ways a long run
fails. The out-of-the-box torch error (`RuntimeError: CUDA out of
memory. Tried to allocate N MiB ...`) is not terribly useful — it
doesn't tell the user what to change.

This module wraps the boundary with:

1. `recommend_grad_accum(peak_bytes, free_bytes, current)` — computes
   the smallest `grad_accum` multiplier that the post-mortem evidence
   says *might* fit. Always doubles at minimum so the next run
   overshoots rather than re-OOMs.
2. `format_oom_message(...)` — renders the user-facing multiline
   message used by the CLI.
3. `catch_cuda_oom(step, current_grad_accum)` — context manager that
   converts a torch.cuda.OutOfMemoryError into a typed `OOMError`
   with filled-in recommendations. Empties the CUDA cache before
   raising so a caller that wants to retry has a clean slate.

The peak-memory numbers come from `torch.cuda.max_memory_allocated()`
which the caller should have been tracking. We don't touch cuda
introspection when cuda isn't available — the context manager is a
no-op on MPS/CPU.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from dlm.train.errors import OOMError


def recommend_grad_accum(
    *,
    peak_bytes: int,
    free_at_start_bytes: int,
    current_grad_accum: int,
) -> int:
    """Return the next `grad_accum` value to try.

    Heuristic: ratio `peak / free`, rounded up. Always at least 2×
    the current (single-step doubling is the standard anti-OOM move;
    anything smaller risks immediate re-OOM).

    If `free_at_start_bytes` is 0 (e.g., introspection failed), we
    default to a simple double.
    """
    if current_grad_accum < 1:
        current_grad_accum = 1
    if free_at_start_bytes <= 0 or peak_bytes <= 0:
        return current_grad_accum * 2

    ratio = peak_bytes / free_at_start_bytes
    # `grad_accum` scales per-step memory roughly linearly. If peak was
    # 2× free, we need at least 2× more accumulation.
    proposed = int(current_grad_accum * max(ratio, 2.0) + 0.999)
    return max(proposed, current_grad_accum * 2)


def format_oom_message(
    *,
    step: int,
    peak_bytes: int,
    free_at_start_bytes: int,
    current_grad_accum: int,
    recommended_grad_accum: int,
) -> str:
    """Render the CLI-facing OOM message."""
    peak_gb = peak_bytes / 1e9
    free_gb = free_at_start_bytes / 1e9
    return (
        f"CUDA OOM at step {step}.\n"
        f"Measured peak: {peak_gb:.1f} GB. Free at start: {free_gb:.1f} GB.\n"
        f"Suggested: set `grad_accum: {recommended_grad_accum}` "
        f"(was {current_grad_accum}), re-run with --resume."
    )


@contextlib.contextmanager
def catch_cuda_oom(  # pragma: no cover
    *,
    step_ref: list[int],
    current_grad_accum: int,
) -> Iterator[None]:
    """Wrap a training step; convert torch OOM into a typed OOMError.

    `step_ref` is a one-element list so the caller can update the
    current step number before entering the `with` block and the
    handler reads the up-to-date value.

    Callers on non-CUDA devices can skip this — re-raising a
    non-CUDA OOM as an `OOMError` would be actively misleading.
    """
    try:
        yield
    except Exception as exc:  # broad on purpose; torch OOM is a subclass
        # Deferred import — torch isn't guaranteed.
        try:
            import torch
        except ImportError:  # pragma: no cover
            raise

        if not isinstance(exc, torch.cuda.OutOfMemoryError):
            raise

        step = step_ref[-1] if step_ref else 0
        peak = int(torch.cuda.max_memory_allocated())
        # `free` is the amount free at the start of the step — we
        # snapshot it when the context was entered by reading what
        # cuda reports available now + peak used.
        free, _total = torch.cuda.mem_get_info()
        free_at_start = int(free) + peak

        recommended = recommend_grad_accum(
            peak_bytes=peak,
            free_at_start_bytes=free_at_start,
            current_grad_accum=current_grad_accum,
        )

        # Free cache so a retry in the same process has a fresh arena.
        torch.cuda.empty_cache()

        raise OOMError(
            step=step,
            peak_bytes=peak,
            free_at_start_bytes=free_at_start,
            current_grad_accum=current_grad_accum,
            recommended_grad_accum=recommended,
        ) from exc
