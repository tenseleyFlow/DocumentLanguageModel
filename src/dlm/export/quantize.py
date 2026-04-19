"""`llama-quantize` subprocess wrapper with timeout + stderr capture.

All subprocess calls in the export pipeline route through this module
so the timeout + stderr-tail + `SubprocessError` wrapping are uniform.

Why a custom wrapper instead of `subprocess.run(check=True)`:

1. `check=True` raises `CalledProcessError` whose `stderr` is only
   populated when `capture_output=True` was also set. Callers forget
   one or the other.
2. CI runners frequently buffer stdout/stderr; a misbehaving quantize
   job can hang for hours without the user knowing. A 10-minute
   timeout per invocation keeps the failure visible.
3. `SubprocessError` carries only the last ~4 KB of stderr so the
   CLI can surface the useful bit without dumping a megabyte of
   progress bars.
"""

from __future__ import annotations

import subprocess  # nosec B404 — export pipeline needs real subprocess
from collections.abc import Sequence
from typing import Final

from dlm.export.errors import SubprocessError

# Generous upper bound — a 7B → Q4_K_M quantize on CI typically runs
# a minute or two; we cap at 10 to catch hangs without surprising
# large-model jobs.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 600.0

# Keep the last ~4 KB of stderr in the error. Enough to show the
# actual error line(s) without pasting an entire progress log into
# the user's terminal.
_STDERR_TAIL_BYTES: Final[int] = 4096


def run_checked(
    cmd: Sequence[str],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run `cmd`; raise `SubprocessError` on non-zero exit or timeout.

    Captures stdout+stderr as bytes; the caller (`runner.py`) logs
    successful stdout to the export log if it wants. Stderr is
    returned verbatim on success, and as a tail-only slice on failure.
    """
    try:
        proc = subprocess.run(  # nosec B603 — caller-controlled argv only
            list(cmd),
            capture_output=True,
            check=False,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        stderr_tail = _tail_bytes(exc.stderr or b"")
        raise SubprocessError(
            cmd=list(cmd),
            returncode=None,
            stderr_tail=stderr_tail,
        ) from exc

    if proc.returncode != 0:
        stderr_tail = _tail_bytes(proc.stderr or b"")
        raise SubprocessError(
            cmd=list(cmd),
            returncode=proc.returncode,
            stderr_tail=stderr_tail,
        )
    return proc


def _tail_bytes(data: bytes) -> str:
    """Decode the last `_STDERR_TAIL_BYTES` of `data` to str, best-effort."""
    if len(data) > _STDERR_TAIL_BYTES:
        data = data[-_STDERR_TAIL_BYTES:]
    return data.decode("utf-8", errors="replace")
