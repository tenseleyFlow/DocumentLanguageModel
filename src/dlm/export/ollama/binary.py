"""Locate the `ollama` binary + enforce a minimum version (audit F16).

`OLLAMA_MIN_VERSION` is a typed constant in code (not a docstring)
so `grep OLLAMA_MIN_VERSION` finds the single source of truth. The
CI job tests exactly this version for smoke passing, plus one minor
version above, plus one below (which must raise).
"""

from __future__ import annotations

import re
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Final

from dlm.export.ollama.errors import (
    OllamaBinaryNotFoundError,
    OllamaVersionError,
)

# Audit F16: ollama 0.4.2 is the first release whose Modelfile grammar
# matches the `TEMPLATE`/`ADAPTER`/`PARAMETER` shape Sprint 11 emits.
# Bumping this requires a matching CI matrix update + grep sweep.
OLLAMA_MIN_VERSION: Final[tuple[int, int, int]] = (0, 4, 2)

# Common Ollama install paths on macOS + Linux. Tried after PATH lookup.
_STANDARD_PATHS: Final[tuple[Path, ...]] = (
    Path("/usr/local/bin/ollama"),
    Path("/opt/homebrew/bin/ollama"),
    Path("/usr/bin/ollama"),
    Path.home() / ".local" / "bin" / "ollama",
)

_VERSION_RE: Final[re.Pattern[str]] = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def locate_ollama(override: Path | None = None) -> Path:
    """Return the path to the `ollama` binary or raise.

    Lookup order:
    1. `override` (test hook; production code never passes).
    2. `shutil.which("ollama")` — PATH.
    3. Known install paths.

    Raises `OllamaBinaryNotFoundError` with the install link when nothing
    resolves.
    """
    if override is not None:
        if override.is_file():
            return override
        raise OllamaBinaryNotFoundError(f"override path {override} does not exist")

    found = shutil.which("ollama")
    if found:
        return Path(found)

    for candidate in _STANDARD_PATHS:
        if candidate.is_file():
            return candidate

    raise OllamaBinaryNotFoundError(
        "`ollama` is not on PATH and was not found at standard install "
        "locations. Install from https://ollama.com/download and retry."
    )


def ollama_version(binary: Path | None = None) -> tuple[int, int, int]:
    """Parse `ollama --version` into `(major, minor, patch)`.

    Accepts both historical formats:
    - `ollama version is 0.4.2`
    - `ollama version 0.4.2`
    - plain `0.4.2`

    Raises `OllamaBinaryNotFoundError` if the binary isn't callable,
    or `OllamaVersionError` on unparseable output.
    """
    path = binary or locate_ollama()
    try:
        result = subprocess.run(  # nosec B603 — caller-controlled path
            [str(path), "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raise OllamaBinaryNotFoundError(f"cannot execute {path}: {exc}") from exc

    # Ollama prints version to stdout on most platforms; some builds
    # print to stderr. Check both.
    blob = (result.stdout or "") + "\n" + (result.stderr or "")
    match = _VERSION_RE.search(blob)
    if not match:
        raise OllamaVersionError(
            detected=(0, 0, 0),
            required=OLLAMA_MIN_VERSION,
        )
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def check_ollama_version(binary: Path | None = None) -> tuple[int, int, int]:
    """Assert `ollama --version >= OLLAMA_MIN_VERSION` and return the parsed tuple."""
    detected = ollama_version(binary)
    if detected < OLLAMA_MIN_VERSION:
        raise OllamaVersionError(
            detected=detected,
            required=OLLAMA_MIN_VERSION,
        )
    return detected
