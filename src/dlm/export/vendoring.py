"""Resolve paths into the vendored `vendor/llama.cpp/` submodule.

The submodule is pinned via git at a release tag; `scripts/bump-llama-cpp.sh`
handles deliberate bumps. This module does NOT build anything — it
just locates files the Sprint 11 runner hands to `subprocess.run`.

Three primary artifacts:

- `convert_hf_to_gguf.py` — Python script that reads an HF checkpoint
  and writes a GGUF file. Invoked with `sys.executable`.
- `convert_lora_to_gguf.py` — sibling script for PEFT adapters.
- `llama-quantize` — compiled binary (built by cmake). Converts an
  fp16 GGUF into one of the quant levels.

Missing or unbuilt artifacts raise `VendoringError` with a remediation
pointing at `scripts/bump-llama-cpp.sh`. The runner catches + reworks
the message for the CLI; test code can catch the bare typed error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dlm.export.errors import VendoringError

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
VENDOR_LLAMA_CPP: Final[Path] = _REPO_ROOT / "vendor" / "llama.cpp"

CONVERT_HF_TO_GGUF: Final[str] = "convert_hf_to_gguf.py"
CONVERT_LORA_TO_GGUF: Final[str] = "convert_lora_to_gguf.py"
# `llama-quantize` is the binary produced by `cmake --build` of llama.cpp;
# release layouts expose it under `build/bin/` or `bin/` depending on version.
_LLAMA_QUANTIZE_CANDIDATES: Final[tuple[str, ...]] = (
    "build/bin/llama-quantize",
    "bin/llama-quantize",
    "llama-quantize",
    # Legacy pre-rename binary — support during the bump window.
    "build/bin/quantize",
    "bin/quantize",
    "quantize",
)


def llama_cpp_root(override: Path | None = None) -> Path:
    """Return the path to the vendored `llama.cpp` clone.

    `override` is a test hook — production code never passes it.
    Raises `VendoringError` if the directory is missing OR is empty
    (an uninitialized submodule).
    """
    root = override or VENDOR_LLAMA_CPP
    if not root.is_dir():
        raise VendoringError(
            f"vendor/llama.cpp is missing at {root}. "
            "Run `git submodule update --init --recursive` and then "
            "`scripts/bump-llama-cpp.sh build` to materialize the toolchain."
        )
    # An empty dir (uninitialized submodule) is the most common failure.
    try:
        any_entry = next(root.iterdir(), None)
    except OSError as exc:
        raise VendoringError(f"cannot enumerate {root}: {exc}") from exc
    if any_entry is None:
        raise VendoringError(
            f"vendor/llama.cpp is empty at {root}. Run `git submodule update --init --recursive`."
        )
    return root


def convert_hf_to_gguf_py(override: Path | None = None) -> Path:
    """Path to `convert_hf_to_gguf.py` inside the vendored tree."""
    return _resolve_script(CONVERT_HF_TO_GGUF, override)


def convert_lora_to_gguf_py(override: Path | None = None) -> Path:
    """Path to `convert_lora_to_gguf.py` inside the vendored tree."""
    return _resolve_script(CONVERT_LORA_TO_GGUF, override)


def llama_quantize_bin(override: Path | None = None) -> Path:
    """Path to the `llama-quantize` binary.

    Checks several known build-layout locations since llama.cpp's
    build output has moved between releases. If none of the
    candidates exist, `VendoringError` points at the bump script's
    build step.
    """
    root = llama_cpp_root(override)
    for candidate in _LLAMA_QUANTIZE_CANDIDATES:
        path = root / candidate
        if path.is_file():
            return path
    raise VendoringError(
        f"llama-quantize binary not found under {root}. "
        "Run `scripts/bump-llama-cpp.sh build` to compile it."
    )


def pinned_tag(override: Path | None = None) -> str | None:
    """Best-effort read of the pinned llama.cpp tag or short SHA.

    First tries `vendor/llama.cpp/VERSION` (written by the bump script
    as a single-line file); falls back to reading the submodule's
    `HEAD` sha via `.git` file. Returns `None` if neither resolves —
    callers (the export manifest) record `None` as "unknown upstream
    version" rather than refuse the export.
    """
    root = override or VENDOR_LLAMA_CPP
    if not root.is_dir():
        return None

    version_file = root / "VERSION"
    if version_file.is_file():
        content = version_file.read_text(encoding="utf-8").strip()
        if content:
            return content

    # Git submodules carry a `.git` file (not dir) pointing at the
    # parent's `.git/modules/<name>/HEAD`; reading it directly is
    # brittle across git versions. We shell out only if the simpler
    # VERSION file isn't there — Sprint 11's bump script maintains it.
    return None


def _resolve_script(name: str, override: Path | None) -> Path:
    root = llama_cpp_root(override)
    path = root / name
    if not path.is_file():
        raise VendoringError(
            f"{name} not found under {root}. "
            "Your vendored llama.cpp may be pre-{name} era; "
            "run `scripts/bump-llama-cpp.sh` to update."
        )
    return path
