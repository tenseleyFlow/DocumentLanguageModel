"""Resolve paths into the vendored `vendor/llama.cpp/` submodule.

The submodule is pinned via git at a release tag; `scripts/bump-llama-cpp.sh`
handles deliberate bumps. This module does NOT build anything — it
just locates files the export runner hands to `subprocess.run`.

Three primary artifacts:

- `convert_hf_to_gguf.py` — Python script that reads an HF checkpoint
  and writes a GGUF file. Invoked with `sys.executable`.
- `convert_lora_to_gguf.py` — sibling script for PEFT adapters.
- `llama-quantize` — compiled binary (built by cmake). Converts an
  fp16 GGUF into one of the quant levels.
- `llama-server` — compiled binary for the OpenAI-compatible HTTP
  server target added in Sprint 41.

Lookup order for the llama.cpp source tree (convert scripts):

1. `DLM_LLAMA_CPP_ROOT` env var — set by the Homebrew formula so
   `brew install dlm` points at `libexec/vendor/llama.cpp/` without
   needing an in-tree submodule.
2. `vendor/llama.cpp/` relative to the repo root — dev path.

Binary lookup falls through to `shutil.which()` when the vendored
`build/bin/` isn't present, so `brew install llama.cpp`'s
`/opt/homebrew/bin/llama-quantize` satisfies the resolver on brew
installs.

Missing or unbuilt artifacts raise `VendoringError` with a remediation
pointing at `scripts/bump-llama-cpp.sh` (source install) or
`brew install llama.cpp` (brew install). The runner catches + reworks
the message for the CLI.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Final

from dlm.export.errors import VendoringError

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
VENDOR_LLAMA_CPP: Final[Path] = _REPO_ROOT / "vendor" / "llama.cpp"
_ENV_VAR: Final[str] = "DLM_LLAMA_CPP_ROOT"
_BUILD_ENV_VAR: Final[str] = "DLM_LLAMA_CPP_BUILD"
"""When set, `_resolve_binary` checks
`<DLM_LLAMA_CPP_BUILD>/bin/<name>` before the default vendor layout.
Lets users point `dlm export` at a HIP-built llama.cpp without
rebuilding the vendor dir itself (see docs/hardware/rocm.md)."""

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
# `llama-imatrix` produces the importance matrix we feed to quantize for
# k-quant calibration. Same build layout as llama-quantize.
_LLAMA_IMATRIX_CANDIDATES: Final[tuple[str, ...]] = (
    "build/bin/llama-imatrix",
    "bin/llama-imatrix",
    "llama-imatrix",
    # Legacy pre-rename (pre-llama.cpp b3600-ish).
    "build/bin/imatrix",
    "bin/imatrix",
    "imatrix",
)
_LLAMA_SERVER_CANDIDATES: Final[tuple[str, ...]] = (
    "build/bin/llama-server",
    "bin/llama-server",
    "llama-server",
    # Legacy pre-rename binary.
    "build/bin/server",
    "bin/server",
    "server",
)


def llama_cpp_root(override: Path | None = None) -> Path:
    """Return the path to the llama.cpp source tree.

    Resolution order:

    1. `override` kwarg (test hook; production code never passes it).
    2. `$DLM_LLAMA_CPP_ROOT` env var (set by the Homebrew formula).
    3. `vendor/llama.cpp/` at the repo root (source / dev install).

    Raises `VendoringError` if none of those resolve to a non-empty
    directory.
    """
    if override is not None:
        root = override
    elif env_override := os.environ.get(_ENV_VAR):
        root = Path(env_override)
    else:
        root = VENDOR_LLAMA_CPP
    if not root.is_dir():
        raise VendoringError(
            f"llama.cpp source tree missing at {root}. For source installs, "
            "run `git submodule update --init --recursive` and then "
            "`scripts/bump-llama-cpp.sh build`. For brew installs, "
            f"ensure the {_ENV_VAR} env var points at a populated tree "
            "(normally handled by the dlm formula)."
        )
    # An empty dir (uninitialized submodule) is the most common failure.
    try:
        any_entry = next(root.iterdir(), None)
    except OSError as exc:
        raise VendoringError(f"cannot enumerate {root}: {exc}") from exc
    if any_entry is None:
        raise VendoringError(
            f"llama.cpp source tree is empty at {root}. "
            "Run `git submodule update --init --recursive` (source install) "
            f"or unset {_ENV_VAR} and reinstall the dlm formula."
        )
    return root


def convert_hf_to_gguf_py(override: Path | None = None) -> Path:
    """Path to `convert_hf_to_gguf.py` inside the vendored tree."""
    return _resolve_script(CONVERT_HF_TO_GGUF, override)


def convert_lora_to_gguf_py(override: Path | None = None) -> Path:
    """Path to `convert_lora_to_gguf.py` inside the vendored tree."""
    return _resolve_script(CONVERT_LORA_TO_GGUF, override)


def _resolve_binary(
    *,
    name: str,
    candidates: tuple[str, ...],
    override: Path | None,
) -> Path:
    """Find a llama.cpp binary, preferring the vendored build tree then $PATH.

    When `override` is None and the env/vendor tree lacks a compiled
    binary, fall back to `shutil.which(name)` — covers the common
    `brew install llama.cpp` case where the binary lives under
    `/opt/homebrew/bin/`.

    `$DLM_LLAMA_CPP_BUILD`, when set, is checked BEFORE
    the default vendor tree. Lets ROCm users point at the HIP build
    dir produced by `scripts/build-llama-cpp-rocm.sh` without
    clobbering the CPU build.
    """
    if override is None:
        build_env = os.environ.get(_BUILD_ENV_VAR)
        if build_env:
            build_root = Path(build_env)
            for candidate in candidates:
                path = build_root / candidate
                if path.is_file():
                    return path
    root = llama_cpp_root(override)
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    # Fall through to PATH lookup (brew-installed llama.cpp).
    on_path = shutil.which(name)
    if on_path is not None:
        return Path(on_path)
    raise VendoringError(
        f"{name} binary not found under {root} and not on $PATH. For "
        "source installs, run `scripts/bump-llama-cpp.sh build`. For "
        "brew installs, `brew install llama.cpp`."
    )


def llama_quantize_bin(override: Path | None = None) -> Path:
    """Path to the `llama-quantize` binary.

    Checks several known build-layout locations, then falls back to
    `$PATH` — covers both the vendored `build/bin/llama-quantize`
    (source install) and the brew `/opt/homebrew/bin/llama-quantize`
    (brew install with `depends_on "llama.cpp"`).
    """
    return _resolve_binary(
        name="llama-quantize",
        candidates=_LLAMA_QUANTIZE_CANDIDATES,
        override=override,
    )


def llama_imatrix_bin(override: Path | None = None) -> Path:
    """Path to the `llama-imatrix` binary.

    Same resolver shape as `llama_quantize_bin` — checks vendored
    build layouts, then `$PATH`.
    """
    return _resolve_binary(
        name="llama-imatrix",
        candidates=_LLAMA_IMATRIX_CANDIDATES,
        override=override,
    )


def llama_server_bin(override: Path | None = None) -> Path:
    """Path to the `llama-server` binary.

    Same resolver shape as `llama_quantize_bin` — vendored build
    layouts first, then `$PATH`.
    """
    return _resolve_binary(
        name="llama-server",
        candidates=_LLAMA_SERVER_CANDIDATES,
        override=override,
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
    # VERSION file isn't there — the vendored build metadata usually
    # maintains it.
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
