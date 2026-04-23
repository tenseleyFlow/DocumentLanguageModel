"""Export target registry."""

from __future__ import annotations

from dlm.export.errors import UnknownExportTargetError
from dlm.export.targets.base import ExportTarget, SmokeResult, TargetResult
from dlm.export.targets.llama_server import LLAMA_SERVER_TARGET, prepare_llama_server_export
from dlm.export.targets.ollama import OLLAMA_TARGET
from dlm.export.targets.vllm import VLLM_TARGET, finalize_vllm_export, prepare_vllm_export

TARGETS: dict[str, ExportTarget] = {
    OLLAMA_TARGET.name: OLLAMA_TARGET,
    LLAMA_SERVER_TARGET.name: LLAMA_SERVER_TARGET,
    VLLM_TARGET.name: VLLM_TARGET,
}


def available_targets() -> tuple[str, ...]:
    """Return stable export-target names in display order."""
    return tuple(TARGETS.keys())


def resolve_target(name: str) -> ExportTarget:
    """Resolve one configured export target by name."""
    try:
        return TARGETS[name]
    except KeyError as exc:
        raise UnknownExportTargetError(name, available=available_targets()) from exc


__all__ = [
    "ExportTarget",
    "LLAMA_SERVER_TARGET",
    "SmokeResult",
    "TARGETS",
    "TargetResult",
    "VLLM_TARGET",
    "available_targets",
    "finalize_vllm_export",
    "prepare_llama_server_export",
    "prepare_vllm_export",
    "resolve_target",
]
