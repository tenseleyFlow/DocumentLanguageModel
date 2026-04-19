"""Convert a PEFT adapter checkpoint to `adapter.gguf`.

llama.cpp's `convert_lora_to_gguf.py` takes the adapter directory as
a **positional** argument and reads `adapter_config.json` to discover
the base model. Earlier drafts (pre-audit F09) invented a `--base`
flag; the upstream CLI does not have one. This module assembles the
exact upstream signature so a snapshot test against `--help` catches
drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dlm.export import vendoring


def build_convert_lora_args(
    adapter_dir: Path,
    *,
    out_gguf: Path,
    outtype: str = "f16",
    script_override: Path | None = None,
    python_exe: str | None = None,
) -> list[str]:
    """Assemble the `python convert_lora_to_gguf.py <adapter> ...` argv.

    Audit F09: the signature must match the pinned upstream CLI. A
    snapshot test under `tests/unit/export/test_subprocess_args.py`
    diffs this against the expected form and fails on drift.
    """
    script = vendoring.convert_lora_to_gguf_py(script_override)
    return [
        python_exe or sys.executable,
        str(script),
        str(adapter_dir),
        "--outfile",
        str(out_gguf),
        "--outtype",
        outtype,
    ]
