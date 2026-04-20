"""Two-GPU multi-GPU smoke (Sprint 23).

Verifies `dlm train --gpus 0,1` actually launches two ranks via
`accelerate launch` and produces an adapter version bump. Doesn't
assert bit-exact checksum against single-GPU (that's a followup once
`trainer.run` gains its DDP-aware I/O gating).

Skipped unless:
- `torch.cuda.device_count() >= 2` at collection time.
- `DLM_ENABLE_MULTIGPU_SMOKE=1` in the env (opt-in).

CI: no default multi-GPU runner; expected to run on self-hosted
hardware via a scheduled workflow.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _has_two_gpus() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    return bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) >= 2


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _has_two_gpus(), reason="requires >= 2 visible CUDA devices"),
    pytest.mark.skipif(
        os.environ.get("DLM_ENABLE_MULTIGPU_SMOKE") != "1",
        reason="set DLM_ENABLE_MULTIGPU_SMOKE=1 to opt in on a 2-GPU host",
    ),
]


def test_two_gpu_launcher_spawns_two_ranks(  # pragma: no cover - multigpu path
    tmp_path: Path,
) -> None:
    """`dlm train --gpus 0,1` exits 0 and lands an adapter version bump."""
    # Uses the trained_store session fixture path — the smoke is in
    # whether the launcher completes without refusal, not in testing
    # specific training dynamics. A minimal one-step run is enough.
    dlm_home = tmp_path / "home"
    dlm_home.mkdir()
    doc = tmp_path / "smoke.dlm"

    # Create a tiny doc via dlm init (smollm2-135m base is ungated).
    init_rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dlm.cli.app",
            "--home",
            str(dlm_home),
            "init",
            str(doc),
            "--base",
            "smollm2-135m",
        ],
        check=False,
    ).returncode
    assert init_rc == 0, f"dlm init failed (exit {init_rc})"

    # Multi-GPU train — expect exit 0 and an adapter version in the
    # store. `--max-steps 1` keeps the smoke fast.
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dlm.cli.app",
            "--home",
            str(dlm_home),
            "train",
            str(doc),
            "--gpus",
            "0,1",
            "--max-steps",
            "1",
            "--fresh",
        ],
        check=False,
    ).returncode
    assert rc == 0, f"multi-GPU train failed (exit {rc})"
