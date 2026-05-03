"""Per-target export orchestration.

Wraps the prepare → smoke → finalize chain for the OpenAI-compat server
targets (vLLM, MLX-serve) so the CLI doesn't repeat the same plumbing
for each. Returns typed results the CLI renders; smoke failure surfaces
as a populated `smoke` field with `ok=False`, leaving `manifest_path`
unset, so the CLI can decide its own exit code without the dispatcher
making control-flow choices.

External-module imports are dotted (e.g. `from dlm.export import
targets as _targets; _targets.prepare_vllm_export(...)`) so test
fixtures that monkeypatch `dlm.export.targets.<name>` resolve at call
time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dlm.export import targets as _targets

if TYPE_CHECKING:
    from pathlib import Path

    from dlm.base_models.schema import BaseModelSpec
    from dlm.export.targets.base import ExportTarget, SmokeResult, TargetResult
    from dlm.store.paths import StorePath


@dataclass(frozen=True)
class VllmExportRequest:
    """Inputs to `run_vllm_target_export`."""

    target: ExportTarget
    store: StorePath
    spec: BaseModelSpec
    served_model_name: str
    training_sequence_len: int | None
    adapter_name: str | None
    adapter_path_override: Path | None
    declared_adapter_names: tuple[str, ...] | None
    adapter_mix: list[tuple[str, float]] | None
    no_smoke: bool


@dataclass(frozen=True)
class MlxServeExportRequest:
    """Inputs to `run_mlx_serve_target_export`."""

    target: ExportTarget
    store: StorePath
    spec: BaseModelSpec
    adapter_name: str | None
    adapter_path_override: Path | None
    declared_adapter_names: tuple[str, ...] | None
    adapter_mix: list[tuple[str, float]] | None
    no_smoke: bool


@dataclass(frozen=True)
class ServerTargetExportResult:
    """Outcome of a server-target export.

    `manifest_path` is `None` when smoke failed (finalize was skipped);
    the CLI surfaces a smoke-failure exit in that case. `smoke` is `None`
    when `--no-smoke` was set.
    """

    prepared: TargetResult
    smoke: SmokeResult | None
    manifest_path: Path | None


def run_vllm_target_export(req: VllmExportRequest) -> ServerTargetExportResult:
    """Stage vLLM artifacts, smoke-test the server, then finalize the manifest."""
    prepared = _targets.prepare_vllm_export(
        store=req.store,
        spec=req.spec,
        served_model_name=req.served_model_name,
        training_sequence_len=req.training_sequence_len,
        adapter_name=req.adapter_name,
        adapter_path_override=req.adapter_path_override,
        declared_adapter_names=req.declared_adapter_names,
    )

    smoke = None if req.no_smoke else req.target.smoke_test(prepared)
    if smoke is not None and not smoke.ok:
        return ServerTargetExportResult(prepared=prepared, smoke=smoke, manifest_path=None)

    manifest_path = _targets.finalize_vllm_export(
        store=req.store,
        spec=req.spec,
        prepared=prepared,
        smoke_output_first_line=None if smoke is None else smoke.detail,
        adapter_name=req.adapter_name,
        adapter_mix=req.adapter_mix,
    )
    return ServerTargetExportResult(prepared=prepared, smoke=smoke, manifest_path=manifest_path)


def run_mlx_serve_target_export(req: MlxServeExportRequest) -> ServerTargetExportResult:
    """Stage MLX-serve artifacts, smoke-test the server, then finalize the manifest."""
    prepared = _targets.prepare_mlx_serve_export(
        store=req.store,
        spec=req.spec,
        adapter_name=req.adapter_name,
        adapter_path_override=req.adapter_path_override,
        declared_adapter_names=req.declared_adapter_names,
    )

    smoke = None if req.no_smoke else req.target.smoke_test(prepared)
    if smoke is not None and not smoke.ok:
        return ServerTargetExportResult(prepared=prepared, smoke=smoke, manifest_path=None)

    manifest_path = _targets.finalize_mlx_serve_export(
        store=req.store,
        spec=req.spec,
        prepared=prepared,
        smoke_output_first_line=None if smoke is None else smoke.detail,
        adapter_name=req.adapter_name,
        adapter_mix=req.adapter_mix,
    )
    return ServerTargetExportResult(prepared=prepared, smoke=smoke, manifest_path=manifest_path)
