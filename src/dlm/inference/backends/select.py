"""Inference-backend selection logic.

Pure-Python decision layer between the CLI flag (`--backend auto|
pytorch|mlx`) and the concrete backend classes. Three rules:

- `pytorch` — always available, works everywhere.
- `mlx` — only on darwin-arm64 with `mlx` + `mlx-lm` installed. A
  non-darwin `--backend mlx` is a hard refusal with a clear message;
  we never attempt the `mlx` import off-platform.
- `auto` — picks MLX on Apple Silicon when available, else PyTorch.

`mlx_available()` does NOT import `mlx`/`mlx_lm` — it probes with
`importlib.util.find_spec`, which checks dist metadata without
executing module code. The actual import happens inside
`MlxBackend.load` on the happy path.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from typing import TYPE_CHECKING, Any, Literal

from dlm.inference.errors import InferenceError

if TYPE_CHECKING:
    from dlm.hardware.capabilities import Capabilities


BackendName = Literal["pytorch", "mlx"]
RequestedBackend = Literal["auto", "pytorch", "mlx"]


class UnsupportedBackendError(InferenceError):
    """Raised when `--backend mlx` is requested on a non-darwin-arm64 host."""


def is_apple_silicon() -> bool:
    """True iff we're on darwin-arm64.

    Hoisted so tests can monkeypatch without touching `platform`
    globals across the suite.
    """
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    """Probe for `mlx` + `mlx_lm` without importing them.

    Returns False immediately off Apple Silicon so test runs on Linux
    CI never see a partial MLX install confuse the selector.
    """
    if not is_apple_silicon():
        return False
    return (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )


def select_backend(requested: RequestedBackend, caps: Capabilities | None = None) -> BackendName:
    """Resolve the backend name from the user's request + host capabilities.

    `caps` is optional — `select_backend` doesn't actually consult it
    today; it's threaded through so future rules (e.g. prefer MLX only
    on MPS-capable hardware rather than any darwin-arm64 box) don't
    require another signature break.

    Rules:

    - `requested == "pytorch"` → always `"pytorch"`.
    - `requested == "mlx"` on darwin-arm64 with mlx installed → `"mlx"`.
    - `requested == "mlx"` off-platform or without mlx →
      `UnsupportedBackendError` with the exact missing piece.
    - `requested == "auto"`:
      - `mlx_available()` → `"mlx"`
      - else → `"pytorch"`
    """
    _ = caps  # reserved for future hardware-aware selection
    if requested == "pytorch":
        return "pytorch"
    if requested == "mlx":
        if not is_apple_silicon():
            raise UnsupportedBackendError(
                "--backend mlx requires Apple Silicon (darwin-arm64); "
                f"this host reports sys.platform={sys.platform!r}, "
                f"machine={platform.machine()!r}."
            )
        if not mlx_available():
            raise UnsupportedBackendError(
                "--backend mlx requires the mlx extra to be installed; "
                "run `uv sync --extra mlx` and re-try."
            )
        return "mlx"
    # requested == "auto"
    if mlx_available():
        return "mlx"
    return "pytorch"


def build_backend(name: BackendName, caps: Any) -> Any:
    """Instantiate the concrete backend class for `name`.

    Lazy imports so the PyTorch-only path stays free of `mlx` touches,
    and vice versa. The returned object satisfies `InferenceBackend`.
    """
    if name == "pytorch":
        from dlm.inference.backends.pytorch_backend import PyTorchBackend

        return PyTorchBackend(caps)
    if name == "mlx":
        from dlm.inference.backends.mlx_backend import MlxBackend

        return MlxBackend(caps)
    raise ValueError(f"unknown backend name: {name!r}")
