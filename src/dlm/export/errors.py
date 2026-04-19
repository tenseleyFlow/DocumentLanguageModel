"""Typed errors for the GGUF export pipeline.

The export path shells out to llama.cpp's `convert_hf_to_gguf.py`,
`convert_lora_to_gguf.py`, and `llama-quantize`. Each subprocess has
its own failure mode; we wrap them in typed errors so the CLI can
surface the right remediation text without scraping stderr.
"""

from __future__ import annotations


class ExportError(Exception):
    """Base for `dlm.export` errors."""


class VendoringError(ExportError):
    """`vendor/llama.cpp` submodule is missing, uninitialized, or unbuilt.

    Typical remediation: `git submodule update --init --recursive` plus
    a build step. The message should point the user at
    `scripts/bump-llama-cpp.sh` for the canonical setup.
    """


class PreflightError(ExportError):
    """A compatibility probe failed before any subprocess launched.

    Carries the offending probe name + detail so the CLI can print
    actionable text. Unlike `VendoringError`, the remedy is data
    (pick a different quant, fix the tokenizer) rather than tooling.
    """

    def __init__(self, probe: str, detail: str) -> None:
        super().__init__(f"preflight failed [{probe}]: {detail}")
        self.probe = probe
        self.detail = detail


class UnsafeMergeError(ExportError):
    """`--merged` on a QLoRA adapter without `--dequantize` (pitfall #3).

    Merging LoRA deltas into a 4-bit-quantized base loses precision
    silently; we refuse until the user confirms. Message includes the
    exact flag to pass to proceed safely.
    """


class SubprocessError(ExportError):
    """A vendored tool exited non-zero or timed out.

    Captures `cmd`, `returncode`, and `stderr` tail so the CLI can
    show the user what actually went wrong without dumping the whole
    subprocess output.
    """

    def __init__(
        self,
        *,
        cmd: list[str],
        returncode: int | None,
        stderr_tail: str,
    ) -> None:
        suffix = f" (exit {returncode})" if returncode is not None else " (timed out)"
        super().__init__(f"subprocess failed{suffix}: {cmd[0]!r}\n{stderr_tail}")
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


class ExportManifestError(ExportError):
    """`export_manifest.json` is unreadable, mis-shaped, or checksum drift."""
