"""Convert an HF base model to fp16 GGUF, then quantize.

Two-step pipeline invoked per `(spec, quant)` combination:

1. `convert_hf_to_gguf.py <cached_base> --outfile <fp16_gguf> --outtype f16`
2. `llama-quantize <fp16_gguf> <quant_gguf> <QUANT>`

Caching: if `base.<quant>.gguf` already exists in the target dir and
its sha matches the manifest's recorded hash, skip both steps. This is
the "re-running export with unchanged adapter + quant is a no-op"
DoD item.

This module exposes two functions — `build_convert_args` and
`build_quantize_args` — that return the `subprocess.run` argv lists.
The real runner (`runner.py`) consumes them, adds timeout + stderr
capture, and retries. Keeping command assembly as a pure function
makes the snapshot test trivial (audit F09: the `--help` diff gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dlm.export import vendoring
from dlm.export.plan import QuantLevel


def build_convert_hf_args(
    cached_base_dir: Path,
    *,
    out_fp16: Path,
    outtype: str = "f16",
    script_override: Path | None = None,
    python_exe: str | None = None,
) -> list[str]:
    """Assemble the `python convert_hf_to_gguf.py ...` argv.

    Pure string-manipulation; no subprocess, no HF, no FS side effects
    beyond `vendoring.convert_hf_to_gguf_py` resolving its path.
    Test target: snapshot against the pinned upstream CLI surface.
    """
    script = vendoring.convert_hf_to_gguf_py(script_override)
    return [
        python_exe or sys.executable,
        str(script),
        str(cached_base_dir),
        "--outfile",
        str(out_fp16),
        "--outtype",
        outtype,
    ]


def build_quantize_args(
    fp16_gguf: Path,
    *,
    out_quant: Path,
    quant: QuantLevel,
    bin_override: Path | None = None,
    imatrix_path: Path | None = None,
) -> list[str]:
    """Assemble the `llama-quantize <in> <out> <QUANT>` argv.

    Note: `llama-quantize` takes the quant string as a POSITIONAL
    argument (no `--quant` flag upstream). Pass it verbatim.

    `imatrix_path` threads Sprint 11.6's importance-matrix flag in as
    `--imatrix <path>` before the positional arguments. The upstream
    tool ignores imatrix on non-k-quant levels (`Q8_0`, `F16`) so we
    don't branch here — callers decide whether to pass a path at all.
    """
    binary = vendoring.llama_quantize_bin(bin_override)
    argv: list[str] = [str(binary)]
    if imatrix_path is not None:
        argv.extend(["--imatrix", str(imatrix_path)])
    argv.extend([str(fp16_gguf), str(out_quant), quant])
    return argv
