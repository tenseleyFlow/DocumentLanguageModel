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
import shutil
import subprocess
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
    """`dlm train --gpus 0,1` exits 0 and lands an adapter version bump.

    Audit-08 B2: we assert **side effects**, not just exit codes. A
    silent no-op subprocess would pass exit=0 but not create the .dlm
    file or bump any adapter version.
    """
    dlm_home = tmp_path / "home"
    dlm_home.mkdir()
    doc = tmp_path / "smoke.dlm"

    # Create a tiny doc via dlm init (smollm2-135m base is ungated).
    # Use the installed `dlm` entrypoint so the subprocess path can't
    # silently no-op if `python -m dlm.cli.app` ever loses its
    # __main__ block again.
    dlm_bin = shutil.which("dlm")
    assert dlm_bin is not None, "`dlm` entrypoint not on PATH — `uv sync` required"
    init_rc = subprocess.run(
        [
            dlm_bin,
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
    assert doc.exists(), "dlm init exited 0 but did not write the .dlm file"

    # Multi-GPU train — expect exit 0 AND a committed adapter version.
    rc = subprocess.run(
        [
            dlm_bin,
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

    # Side-effect assertion: the adapter version pointer must exist.
    # Without this the assertion above would pass on a silent no-op.
    store_root = dlm_home / "store"
    assert store_root.exists(), "dlm train exited 0 but wrote no store"
    dlm_ids = list(store_root.iterdir())
    assert dlm_ids, "store root is empty — train produced no adapter"
    adapter_pointer = dlm_ids[0] / "adapter" / "current.txt"
    assert adapter_pointer.exists(), (
        f"no adapter/current.txt under {dlm_ids[0]} — multi-GPU train "
        f"subprocess exited 0 without committing an adapter version"
    )
