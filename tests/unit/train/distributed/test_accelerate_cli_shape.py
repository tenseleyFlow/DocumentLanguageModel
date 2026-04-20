"""Snapshot test: `accelerate launch --help` carries the flags we rely on.

The launcher builds a fixed argv: `accelerate launch --num_processes N
--mixed_precision bf16 --machine_rank 0 --num_machines 1 -m
dlm.train.distributed.worker_entry <args>`. If upstream renames any
of those flags (audit F17), this test catches it in CI instead of at
user-invocation time.

Skipped when the `accelerate` executable isn't on PATH; the project's
runtime deps include it, so in the standard dev env this runs.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

_REQUIRED_FLAGS = (
    "--num_processes",
    "--mixed_precision",
    "--machine_rank",
    "--num_machines",
    "-m",
)


pytestmark = pytest.mark.skipif(
    shutil.which("accelerate") is None,
    reason="accelerate CLI not on PATH",
)


def _accelerate_help() -> str:
    result = subprocess.run(
        ["accelerate", "launch", "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout + result.stderr


@pytest.mark.parametrize("flag", _REQUIRED_FLAGS)
def test_flag_present_in_accelerate_help(flag: str) -> None:
    """Each flag `build_accelerate_cmd` emits must still exist upstream."""
    help_text = _accelerate_help()
    assert flag in help_text, (
        f"accelerate launch no longer accepts {flag!r}. "
        "Update `dlm.train.distributed.launcher.build_accelerate_cmd` "
        "to match the new surface."
    )


def test_mixed_precision_accepts_bf16() -> None:
    """Our pinned choice (`bf16`) must still be in the enum."""
    help_text = _accelerate_help()
    # argparse renders `{no,fp16,bf16,fp8}` for the choices.
    assert "bf16" in help_text
