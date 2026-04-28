"""Export-target protocol shared by runtime-specific export surfaces.

"target" is a first-class concept so each registered runtime (Ollama,
vLLM, llama-server, MLX-serve) plugs into the same shape instead of
growing ad-hoc CLI branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dlm.export.dispatch import DispatchResult


TargetName = Literal["ollama", "vllm", "llama-server", "mlx-serve"]


@dataclass(frozen=True)
class SmokeResult:
    """Outcome of a target-specific smoke test."""

    attempted: bool
    ok: bool
    detail: str | None = None


@dataclass(frozen=True)
class TargetResult:
    """Prepared runtime target artifacts."""

    name: str
    export_dir: Path
    manifest_path: Path
    artifacts: tuple[Path, ...] = ()
    launch_script_path: Path | None = None
    config_path: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ExportTarget(Protocol):
    """Shared runtime-target interface."""

    name: str

    def prepare(self, ctx: DispatchResult) -> TargetResult:
        """Stage target-specific artifacts from a dispatch result."""
        ...

    def launch_command(self, prepared: TargetResult) -> list[str]:
        """Return the canonical launch argv for the prepared target."""
        ...

    def smoke_test(self, prepared: TargetResult) -> SmokeResult:
        """Run or summarize the target-specific smoke test."""
        ...
