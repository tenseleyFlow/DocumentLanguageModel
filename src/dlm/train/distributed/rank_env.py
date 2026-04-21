"""Per-rank environment probes — read DDP state from env vars.

`accelerate launch` (and `torchrun`) sets `WORLD_SIZE`, `RANK`, and
`LOCAL_RANK` before spawning each rank's subprocess. We read those
values to thread `world_size` through the planner + lock writer and
to gate I/O on rank 0 inside the trainer — without taking a runtime
dependency on an `Accelerator` instance.

No `accelerate` import happens here; the env vars are the public
contract.
"""

from __future__ import annotations

import os


def detect_world_size() -> int:
    """Return the DDP world_size from env, or 1 in single-process.

    Reads `WORLD_SIZE` (the canonical name used by `accelerate launch`,
    `torchrun`, and `torch.distributed.launch`). Bad / missing /
    zero values degrade to 1 — the single-process contract. A
    malformed value is treated as a configuration error rather than
    silently corrupting downstream math, so we raise rather than
    mask.
    """
    raw = os.environ.get("WORLD_SIZE")
    if raw is None or raw == "":
        return 1
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"WORLD_SIZE env var is not an integer: {raw!r}") from exc
    if value < 1:
        return 1
    return value


def detect_rank() -> int:
    """Return this process's DDP rank (0..world_size-1), or 0 in single-process.

    Reads `RANK` first (multi-node canonical), falls back to
    `LOCAL_RANK` (single-node), finally 0. Same bad-value policy as
    `detect_world_size`.
    """
    for key in ("RANK", "LOCAL_RANK"):
        raw = os.environ.get(key)
        if raw is None or raw == "":
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} env var is not an integer: {raw!r}") from exc
        if value < 0:
            return 0
        return value
    return 0


def is_rank_zero() -> bool:
    """True on rank 0 and in single-process; False on every other rank.

    Preferred over manual `detect_rank() == 0` checks because it also
    covers the "env vars absent" path as "treat as single-process."
    """
    return detect_rank() == 0
