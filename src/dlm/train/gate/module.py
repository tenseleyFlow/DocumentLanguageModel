"""The `Gate` module — small MLP that maps a prompt embedding to
per-adapter weights.

Shape:

    Linear(input_dim → hidden_proj_dim) → GELU
    Linear(hidden_proj_dim → n_adapters)
    Softmax over adapter axis

For a default config (input_dim=2048, hidden=64, n_adapters=4) the
parameter count is:

    (2048 * 64 + 64) + (64 * 4 + 4) = 131,328 params ≈ 0.5 MB fp32

Storage uses safetensors so the file is easy to inspect and round-trips
across torch versions. The accompanying `GateMetadata` records the
adapter name order, input/hidden dims, and the training mode (trained
vs uniform) so inference can rebuild the module without guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch
    from torch import Tensor


@dataclass(frozen=True)
class GateMetadata:
    """Sidecar metadata for a persisted gate.

    `mode="uniform"` skips loading the weights file entirely — the
    inference layer synthesizes a uniform tensor directly. `mode="trained"`
    loads the safetensors payload.
    """

    input_dim: int
    hidden_proj_dim: int
    adapter_names: tuple[str, ...]
    mode: Literal["trained", "uniform"]
    # Entropy-regularization weight the trainer used. Preserved for
    # audit + reproducibility; inference ignores it.
    entropy_lambda: float = 0.01

    @property
    def n_adapters(self) -> int:
        return len(self.adapter_names)

    def to_json(self) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "hidden_proj_dim": self.hidden_proj_dim,
            "adapter_names": list(self.adapter_names),
            "mode": self.mode,
            "entropy_lambda": self.entropy_lambda,
        }

    @classmethod
    def from_json(cls, raw: dict[str, object]) -> GateMetadata:
        from dlm.train.gate.errors import GateConfigError

        required = {"input_dim", "hidden_proj_dim", "adapter_names", "mode"}
        missing = required - set(raw)
        if missing:
            raise GateConfigError(f"gate_config.json missing fields: {sorted(missing)}")
        mode = raw["mode"]
        if mode not in ("trained", "uniform"):
            raise GateConfigError(f"gate_config.json mode must be trained|uniform, got {mode!r}")
        adapter_names = raw["adapter_names"]
        if not isinstance(adapter_names, list) or not all(
            isinstance(n, str) for n in adapter_names
        ):
            raise GateConfigError("gate_config.json adapter_names must be a list of strings")
        input_dim_raw = raw["input_dim"]
        hidden_proj_raw = raw["hidden_proj_dim"]
        if not isinstance(input_dim_raw, int) or not isinstance(hidden_proj_raw, int):
            raise GateConfigError("gate_config.json input_dim/hidden_proj_dim must be integers")
        entropy_raw = raw.get("entropy_lambda", 0.01)
        if not isinstance(entropy_raw, int | float):
            raise GateConfigError("gate_config.json entropy_lambda must be numeric")
        return cls(
            input_dim=input_dim_raw,
            hidden_proj_dim=hidden_proj_raw,
            adapter_names=tuple(adapter_names),
            mode=mode,
            entropy_lambda=float(entropy_raw),
        )


def build_gate(
    *,
    input_dim: int,
    hidden_proj_dim: int,
    n_adapters: int,
) -> Gate:
    """Construct a fresh Gate module on CPU with small-init weights."""
    return Gate(
        input_dim=input_dim,
        hidden_proj_dim=hidden_proj_dim,
        n_adapters=n_adapters,
    )


class Gate:
    """Small MLP: `Linear → GELU → Linear → Softmax`.

    Kept as a thin wrapper so `torch` only imports when the module is
    actually instantiated — schema / config tests never touch CUDA or
    MPS state.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_proj_dim: int,
        n_adapters: int,
    ) -> None:
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_proj_dim < 1:
            raise ValueError(f"hidden_proj_dim must be positive, got {hidden_proj_dim}")
        if n_adapters < 2:
            raise ValueError(f"n_adapters must be >= 2, got {n_adapters}")
        from torch import nn

        self.input_dim = input_dim
        self.hidden_proj_dim = hidden_proj_dim
        self.n_adapters = n_adapters
        # Reuse the parameter names safetensors expects so save/load is
        # round-trip clean without a custom key-rewrite layer.
        self._mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_proj_dim),
            nn.GELU(),
            nn.Linear(hidden_proj_dim, n_adapters),
        )

    @property
    def module(self) -> torch.nn.Module:
        """The underlying `nn.Module`. Accessors expose it for
        `.parameters()`, `.to(device)`, etc."""
        return self._mlp

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self._mlp.parameters())

    def forward(self, x: Tensor) -> Tensor:
        """Map `x: (..., input_dim)` to softmax weights `(..., n_adapters)`."""
        from torch.nn import functional as F  # noqa: N812 — torch convention

        logits = self._mlp(x)
        return F.softmax(logits, dim=-1)

    __call__ = forward
