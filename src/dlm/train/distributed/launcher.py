"""`accelerate launch` subprocess wrapper.

Single entry point: `launch_multi_gpu(device_ids, cli_args)`. Spawns
`accelerate launch --num_processes N -m dlm.train.distributed.worker_entry
<cli_args...>`. `CUDA_VISIBLE_DEVICES` is set on the child environment
so the ranks see exactly the requested device ids.

The command-builder is split out (`build_accelerate_cmd`) so tests
can assert the invocation shape without actually running a
subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_accelerate_cmd(
    device_ids: Sequence[int],
    cli_args: Sequence[str],
    *,
    mixed_precision: str = "bf16",
) -> list[str]:
    """Return the `accelerate launch` argv for `num_processes=len(device_ids)`.

    Pinned flags:
    - `--num_processes len(device_ids)`
    - `--mixed_precision <mp>` (default bf16; fp16 is the alternate)
    - `--machine_rank 0 --num_machines 1` (single-node only)
    - `-m dlm.train.distributed.worker_entry` — the `-m` target

    `cli_args` are forwarded verbatim to the worker; typically the
    same argv `dlm train` received, minus the `--gpus` flag.
    """
    if len(device_ids) < 1:
        raise ValueError("build_accelerate_cmd: device_ids is empty")
    return [
        "accelerate",
        "launch",
        "--num_processes",
        str(len(device_ids)),
        "--mixed_precision",
        mixed_precision,
        "--machine_rank",
        "0",
        "--num_machines",
        "1",
        "-m",
        "dlm.train.distributed.worker_entry",
        *cli_args,
    ]


def launch_multi_gpu(  # pragma: no cover - subprocess, covered by slow smoke
    device_ids: Sequence[int],
    cli_args: Sequence[str],
    *,
    mixed_precision: str = "bf16",
) -> int:
    """Run `accelerate launch` in a subprocess; return the exit code.

    Sets `CUDA_VISIBLE_DEVICES` so child ranks see exactly the
    requested devices. Propagates stdout/stderr to the parent.
    """
    cmd = build_accelerate_cmd(device_ids, cli_args, mixed_precision=mixed_precision)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
    completed = subprocess.run(
        cmd,
        check=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )
    return int(completed.returncode)
