"""Ollama target wrapper.

The text GGUF export path already owns Modelfile emission, registration,
and smoke testing. This module wraps that behavior in an
`ExportTarget` implementation so other runtimes can slot into a shared
registry without rewriting the existing Ollama flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dlm.export.targets.base import ExportTarget, SmokeResult, TargetResult

if TYPE_CHECKING:
    from dlm.export.dispatch import DispatchResult


class OllamaTarget:
    """Registered export target for the existing Ollama path."""

    name = "ollama"

    def prepare(self, ctx: DispatchResult) -> TargetResult:
        return TargetResult(
            name=self.name,
            export_dir=ctx.export_dir,
            manifest_path=ctx.manifest_path,
            artifacts=tuple(ctx.artifacts),
            extras={"banner_lines": list(ctx.banner_lines)},
        )

    def launch_command(self, prepared: TargetResult) -> list[str]:
        _ = prepared
        return []

    def smoke_test(self, prepared: TargetResult) -> SmokeResult:
        _ = prepared
        return SmokeResult(
            attempted=False,
            ok=True,
            detail="ollama smoke remains managed by the existing export runner",
        )


OLLAMA_TARGET = OllamaTarget()
assert isinstance(OLLAMA_TARGET, ExportTarget)
