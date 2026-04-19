"""`ollama run` wrapper — post-export smoke test.

Default prompt is `"Hello."` — just enough to verify the model loads
and generates one coherent reply. Empty output or non-zero exit
raises `OllamaSmokeError`. The runner appends the first line of
stdout to `manifest.exports[-1].smoke_output_first_line` so the
smoke is auditable from `dlm show` without re-running inference.

`--no-smoke` skips this entirely at the runner level; this module
exists only to be called on the happy path.
"""

from __future__ import annotations

import subprocess  # nosec B404
from pathlib import Path

from dlm.export.ollama.binary import locate_ollama
from dlm.export.ollama.errors import OllamaSmokeError

_DEFAULT_TIMEOUT_SECONDS = 120.0
_DEFAULT_PROMPT = "Hello."


def ollama_run(
    *,
    name: str,
    prompt: str = _DEFAULT_PROMPT,
    binary: Path | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Invoke `ollama run <name> <prompt>`, return stdout.

    Raises `OllamaSmokeError` on:
    - Non-zero exit code.
    - Empty stdout (the model "succeeded" but produced nothing — a
      runaway-stop scenario Sprint 12 is specifically guarding against).
    - Subprocess timeout.
    """
    exe = binary or locate_ollama()
    try:
        result = subprocess.run(  # nosec B603 — caller-controlled argv
            [str(exe), "run", name, prompt],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise OllamaSmokeError(
            stdout=(exc.stdout or b"").decode("utf-8", errors="replace"),
            stderr=(exc.stderr or b"").decode("utf-8", errors="replace")
            + f"\n(timed out after {timeout}s)",
        ) from exc

    if result.returncode != 0:
        raise OllamaSmokeError(stdout=result.stdout, stderr=result.stderr)
    if not result.stdout.strip():
        raise OllamaSmokeError(
            stdout=result.stdout,
            stderr=result.stderr
            + "\n(empty stdout — the model produced no output for the smoke prompt)",
        )
    return result.stdout


def first_line(text: str) -> str:
    """Return the first non-empty line of `text`, truncated to 200 chars.

    Used to stamp `ExportSummary.smoke_output_first_line` without
    bloating the manifest JSON.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return ""
