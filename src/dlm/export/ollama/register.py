"""`ollama create` wrapper serialized by a global file lock (audit F14).

Two `dlm export` processes racing on the same Ollama registry can
corrupt Ollama's internal store (manifests overlap, tags point at
half-written blobs). We serialize via `~/.dlm/ollama.lock` held for
the duration of `ollama create`.

The lock is *global* per user — not per-`.dlm` store — because Ollama's
registry lives at the OS user level, not per-document. Two users on
the same machine running two unrelated `.dlm`s could still step on
each other; Ollama's own ~/.ollama/ directory layout doesn't have
inter-user synchronization, which is outside our scope.

Subprocess: `ollama create <name> -f <modelfile>` run with `cwd` set
to the `exports/<quant>/` directory so relative paths in the
Modelfile (`./base.gguf`, `./adapter.gguf`) resolve. 10-minute timeout.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404
from pathlib import Path

from dlm.export.ollama.binary import locate_ollama
from dlm.export.ollama.errors import OllamaCreateError
from dlm.store.lock import exclusive

_DEFAULT_TIMEOUT_SECONDS = 600.0
_DEFAULT_LOCK_TIMEOUT_SECONDS = 120.0


def ollama_lock_path(dlm_home: Path | None = None) -> Path:
    """Return `~/.dlm/ollama.lock` (override via `dlm_home` for tests).

    The lock lives at user-scope, not inside any particular store —
    this is what serializes `dlm export` runs across unrelated `.dlm`
    files that happen to write to the same Ollama registry.
    """
    root = dlm_home or Path(os.environ.get("DLM_HOME") or (Path.home() / ".dlm"))
    return root / "ollama.lock"


def ollama_create(
    *,
    name: str,
    modelfile_path: Path,
    cwd: Path,
    binary: Path | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    lock_timeout: float = _DEFAULT_LOCK_TIMEOUT_SECONDS,
    dlm_home: Path | None = None,
) -> str:
    """Run `ollama create <name> -f <modelfile>` under an exclusive lock.

    Returns the subprocess stdout on success; raises `OllamaCreateError`
    with captured stdout+stderr on non-zero exit or timeout.

    `binary` / `dlm_home` are test hooks.
    """
    exe = binary or locate_ollama()
    lock_path = ollama_lock_path(dlm_home)

    with exclusive(lock_path, timeout=lock_timeout):
        try:
            result = subprocess.run(  # nosec B603 — caller-controlled argv
                [str(exe), "create", name, "-f", str(modelfile_path)],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise OllamaCreateError(
                stdout=(exc.stdout or b"").decode("utf-8", errors="replace"),
                stderr=(exc.stderr or b"").decode("utf-8", errors="replace")
                + f"\n(timed out after {timeout}s)",
            ) from exc

    if result.returncode != 0:
        raise OllamaCreateError(stdout=result.stdout, stderr=result.stderr)
    return result.stdout
