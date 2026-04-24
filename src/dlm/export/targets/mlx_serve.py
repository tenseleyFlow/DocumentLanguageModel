"""MLX HTTP server target helpers."""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path

from dlm.base_models import BaseModelSpec
from dlm.export.errors import ExportError, TargetSmokeError
from dlm.export.manifest import ExportManifest, build_artifact, save_export_manifest, utc_now
from dlm.export.record import append_export_summary
from dlm.export.smoke import smoke_openai_compat_server
from dlm.export.targets.base import ExportTarget, SmokeResult, TargetResult
from dlm.inference.backends.mlx_backend import stage_mlx_adapter_dir
from dlm.inference.backends.select import is_apple_silicon, mlx_available
from dlm.io.atomic import write_text
from dlm.store.paths import StorePath

MLX_SERVE_EXPORT_SUBDIR = "mlx-serve"
LAUNCH_SCRIPT_FILENAME = "mlx_serve_launch.sh"
_HF_QUANT = "hf"
_DEFAULT_ADAPTER_DIRNAME = "adapter"
_MIXED_ADAPTER_DIRNAME = "mixed"


class MlxServeTarget:
    """Registered export target for MLX HTTP server launch artifacts."""

    name = "mlx-serve"

    def prepare(self, ctx: object) -> TargetResult:
        raise NotImplementedError("mlx-serve exports are prepared via prepare_mlx_serve_export()")

    def launch_command(self, prepared: TargetResult) -> list[str]:
        return _build_command(prepared, use_script_dir=True)

    def smoke_test(self, prepared: TargetResult) -> SmokeResult:
        try:
            first_line = smoke_openai_compat_server(_build_command(prepared, use_script_dir=False))
        except (OSError, TargetSmokeError, ExportError) as exc:
            return SmokeResult(attempted=True, ok=False, detail=str(exc))
        return SmokeResult(attempted=True, ok=True, detail=first_line)


def prepare_mlx_serve_export(
    *,
    store: StorePath,
    spec: BaseModelSpec,
    adapter_name: str | None,
    adapter_path_override: Path | None,
    declared_adapter_names: tuple[str, ...] | None,
) -> TargetResult:
    """Stage an MLX-loadable adapter dir plus launch script."""

    _require_mlx_runtime()
    source_adapter_dir, staged_dirname, adapter_version = _resolve_source_adapter(
        store=store,
        adapter_name=adapter_name,
        adapter_path_override=adapter_path_override,
        declared_adapter_names=declared_adapter_names,
    )

    export_dir = store.exports / MLX_SERVE_EXPORT_SUBDIR
    export_dir.mkdir(parents=True, exist_ok=True)

    staged_adapter_dir = export_dir / staged_dirname
    if staged_adapter_dir.exists():
        shutil.rmtree(staged_adapter_dir)
    stage_mlx_adapter_dir(source_adapter_dir, staged_adapter_dir, base_hf_id=spec.hf_id)

    launch_script_path = export_dir / LAUNCH_SCRIPT_FILENAME
    draft = TargetResult(
        name=MLX_SERVE_TARGET.name,
        export_dir=export_dir,
        manifest_path=export_dir / "export_manifest.json",
        artifacts=(),
        launch_script_path=launch_script_path,
        extras={
            "model": spec.hf_id,
            "adapter_dir": staged_adapter_dir,
            "adapter_version": adapter_version,
        },
    )
    write_text(launch_script_path, _render_launch_script(MLX_SERVE_TARGET.launch_command(draft)))
    launch_script_path.chmod(0o755)
    return TargetResult(
        name=draft.name,
        export_dir=draft.export_dir,
        manifest_path=draft.manifest_path,
        artifacts=tuple(_artifact_paths(export_dir)),
        launch_script_path=draft.launch_script_path,
        config_path=None,
        extras=draft.extras,
    )


def finalize_mlx_serve_export(
    *,
    store: StorePath,
    spec: BaseModelSpec,
    prepared: TargetResult,
    smoke_output_first_line: str | None,
    adapter_name: str | None,
    adapter_mix: list[tuple[str, float]] | None,
) -> Path:
    """Write export_manifest.json and append the store export summary."""

    from dlm import __version__ as dlm_version

    artifacts = [
        build_artifact(prepared.export_dir, path) for path in _artifact_paths(prepared.export_dir)
    ]
    adapter_version = _require_prepared_int(prepared, "adapter_version")
    manifest = ExportManifest(
        target=MLX_SERVE_TARGET.name,
        quant=_HF_QUANT,
        merged=False,
        dequantized=False,
        ollama_name=None,
        created_at=utc_now(),
        created_by=f"dlm-{dlm_version}",
        llama_cpp_tag=None,
        base_model_hf_id=spec.hf_id,
        base_model_revision=spec.revision,
        adapter_version=adapter_version,
        artifacts=artifacts,
    )
    manifest_path = save_export_manifest(prepared.export_dir, manifest)
    append_export_summary(
        store=store,
        quant=_HF_QUANT,
        merged=False,
        target=MLX_SERVE_TARGET.name,
        llama_cpp_tag=None,
        artifacts=artifacts,
        ollama_name=None,
        ollama_version_str=None,
        smoke_first_line=smoke_output_first_line,
        adapter_name=adapter_name,
        adapter_mix=adapter_mix,
    )
    return manifest_path


def _resolve_source_adapter(
    *,
    store: StorePath,
    adapter_name: str | None,
    adapter_path_override: Path | None,
    declared_adapter_names: tuple[str, ...] | None,
) -> tuple[Path, str, int]:
    if adapter_path_override is not None:
        if not adapter_path_override.exists():
            raise ExportError(f"adapter_path_override {adapter_path_override} does not exist")
        return (
            adapter_path_override,
            _MIXED_ADAPTER_DIRNAME,
            _version_from_dir_name(adapter_path_override),
        )

    if declared_adapter_names and adapter_name is None:
        raise ExportError(
            "mlx-serve exports one adapter at a time; pass `--adapter <name>` "
            "or `--adapter-mix` for multi-adapter documents."
        )

    if adapter_name is not None:
        path = store.resolve_current_adapter_for(adapter_name)
        pointer = store.adapter_current_pointer_for(adapter_name)
        if path is None or not path.exists():
            raise ExportError(
                f"no current adapter under {pointer}; run `dlm train` before exporting."
            )
        return path, adapter_name, _version_from_dir_name(path)

    path = store.resolve_current_adapter()
    pointer = store.adapter_current_pointer
    if path is None or not path.exists():
        raise ExportError(f"no current adapter under {pointer}; run `dlm train` before exporting.")
    return path, _DEFAULT_ADAPTER_DIRNAME, _version_from_dir_name(path)


def _require_mlx_runtime() -> None:
    if not is_apple_silicon():
        raise ExportError(
            "mlx-serve export requires Apple Silicon (darwin-arm64); "
            "this target is not available on CUDA, ROCm, or CPU-only hosts."
        )
    if not mlx_available():
        raise ExportError(
            "mlx-serve export requires the mlx extra to be installed; "
            "run `uv sync --extra mlx` and re-try."
        )


def _artifact_paths(export_dir: Path) -> list[Path]:
    artifacts: list[Path] = []
    for path in sorted(export_dir.rglob("*")):
        if path.is_file() and path.name != "export_manifest.json":
            artifacts.append(path)
    return artifacts


def _build_command(prepared: TargetResult, *, use_script_dir: bool) -> list[str]:
    model = _require_prepared_str(prepared, "model")
    adapter_dir = _require_prepared_path(prepared, "adapter_dir")
    return [
        "python",
        "-m",
        "mlx_lm.server",
        "--model",
        model,
        "--adapter-path",
        _script_dir_arg(adapter_dir) if use_script_dir else str(adapter_dir),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]


def _script_dir_arg(path: Path) -> str:
    return f"$SCRIPT_DIR/{path.name}"


def _render_launch_script(command: list[str]) -> str:
    rendered = " ".join(_quote_script_arg(arg) for arg in command)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
        f'exec {rendered} "$@"\n'
    )


def _quote_script_arg(arg: str) -> str:
    if arg.startswith("$SCRIPT_DIR/"):
        return f'"{arg}"'
    return shlex.quote(arg)


def _version_from_dir_name(path: Path) -> int:
    stem = path.name
    if not stem.startswith("v") or not stem[1:].isdigit():
        return 1
    return int(stem[1:])


def _require_prepared_str(prepared: TargetResult, key: str) -> str:
    value = prepared.extras.get(key)
    if not isinstance(value, str) or not value:
        raise ExportError(f"mlx-serve prepared target missing string extra {key!r}")
    return value


def _require_prepared_path(prepared: TargetResult, key: str) -> Path:
    value = prepared.extras.get(key)
    if not isinstance(value, Path):
        raise ExportError(f"mlx-serve prepared target missing Path extra {key!r}")
    return value


def _require_prepared_int(prepared: TargetResult, key: str) -> int:
    value = prepared.extras.get(key)
    if not isinstance(value, int):
        raise ExportError(f"mlx-serve prepared target missing int extra {key!r}")
    return value


MLX_SERVE_TARGET = MlxServeTarget()
assert isinstance(MLX_SERVE_TARGET, ExportTarget)
