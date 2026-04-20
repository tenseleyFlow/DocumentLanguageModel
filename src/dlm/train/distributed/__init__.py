"""Distributed (multi-GPU) training via HuggingFace Accelerate.

Sprint 23 scope: single-node, multi-GPU DDP. No multi-node, FSDP, or
DeepSpeed.

Surface:

- `GpuSpec` / `parse_gpus` — typed representation of the `--gpus`
  CLI flag (`all` / `N` / `0,1,2`).
- `launch_multi_gpu` — subprocess wrapper that spawns
  `accelerate launch -m dlm.train.distributed.worker_entry`.
- `rank_io.master_only` / `barrier` / `gather_metrics` — per-rank
  helpers used by the worker to keep I/O on rank 0 only.
- `worker_entry` — the `-m` target that runs inside each rank
  subprocess.

Imports are lazy where possible: `accelerate` is a runtime-only
dependency and must not be imported at CLI boot.
"""

from __future__ import annotations

from dlm.train.distributed.gpus import GpuSpec, UnsupportedGpuSpecError, parse_gpus
from dlm.train.distributed.launcher import build_accelerate_cmd, launch_multi_gpu

__all__ = [
    "GpuSpec",
    "UnsupportedGpuSpecError",
    "build_accelerate_cmd",
    "launch_multi_gpu",
    "parse_gpus",
]
