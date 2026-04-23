"""vLLM target helpers."""

from __future__ import annotations

import json
import platform
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from dlm.base_models import BaseModelSpec
from dlm.export.errors import ExportError, TargetSmokeError
from dlm.export.manifest import ExportManifest, build_artifact, save_export_manifest, utc_now
from dlm.export.ollama.modelfile_shared import resolve_num_ctx
from dlm.export.record import append_export_summary
from dlm.export.smoke import smoke_openai_compat_server
from dlm.export.targets.base import ExportTarget, SmokeResult, TargetResult
from dlm.io.atomic import write_text
from dlm.store.paths import StorePath

VLLM_EXPORT_SUBDIR = "vllm"
VLLM_CONFIG_FILENAME = "vllm_config.json"
LAUNCH_SCRIPT_FILENAME = "vllm_launch.sh"
_ADAPTERS_DIRNAME = "adapters"
_HF_QUANT = "hf"
_DEFAULT_MODULE_NAME = "adapter"
_MIXED_MODULE_NAME = "mixed"


@dataclass(frozen=True)
class LoraModule:
    name: str
    path: Path
    adapter_version: int


class VllmTarget:
    """Registered export target for vLLM launch artifacts."""

    name = "vllm"

    def prepare(self, ctx: object) -> TargetResult:
        raise NotImplementedError("vllm exports are prepared via prepare_vllm_export()")

    def launch_command(self, prepared: TargetResult) -> list[str]:
        return _build_command(prepared, use_script_dir=True)

    def smoke_test(self, prepared: TargetResult) -> SmokeResult:
        try:
            first_line = smoke_openai_compat_server(
                _build_command(prepared, use_script_dir=False),
                env=_runtime_env(prepared),
            )
        except (OSError, TargetSmokeError, ExportError) as exc:
            return SmokeResult(attempted=True, ok=False, detail=str(exc))
        return SmokeResult(attempted=True, ok=True, detail=first_line)


def prepare_vllm_export(
    *,
    store: StorePath,
    spec: BaseModelSpec,
    served_model_name: str,
    training_sequence_len: int | None,
    adapter_name: str | None,
    adapter_path_override: Path | None,
    declared_adapter_names: tuple[str, ...] | None,
) -> TargetResult:
    """Stage vLLM launch artifacts plus local adapter module copies."""

    export_dir = store.exports / VLLM_EXPORT_SUBDIR
    export_dir.mkdir(parents=True, exist_ok=True)

    adapters_dir = export_dir / _ADAPTERS_DIRNAME
    if adapters_dir.exists():
        shutil.rmtree(adapters_dir)
    adapters_dir.mkdir(parents=True, exist_ok=True)

    modules = _stage_modules(
        store=store,
        adapters_dir=adapters_dir,
        adapter_name=adapter_name,
        adapter_path_override=adapter_path_override,
        declared_adapter_names=declared_adapter_names,
    )
    if not modules:
        raise ExportError("vllm export needs at least one adapter module")

    context_length = resolve_num_ctx(training_sequence_len, spec.context_length)
    runtime_env = _default_runtime_env()
    config_path = export_dir / VLLM_CONFIG_FILENAME
    launch_script_path = export_dir / LAUNCH_SCRIPT_FILENAME
    draft = TargetResult(
        name=VLLM_TARGET.name,
        export_dir=export_dir,
        manifest_path=export_dir / "export_manifest.json",
        artifacts=(),
        launch_script_path=launch_script_path,
        config_path=config_path,
        extras={
            "model": spec.hf_id,
            "revision": spec.revision,
            "served_model_name": served_model_name,
            "module_specs": tuple(modules),
            "adapter_version": max(module.adapter_version for module in modules),
            "context_length": context_length,
            "runtime_env": runtime_env,
        },
    )
    write_text(config_path, _render_config(draft))
    write_text(
        launch_script_path,
        _render_launch_script(VLLM_TARGET.launch_command(draft), _runtime_env(draft)),
    )
    launch_script_path.chmod(0o755)
    return TargetResult(
        name=draft.name,
        export_dir=draft.export_dir,
        manifest_path=draft.manifest_path,
        artifacts=tuple(_artifact_paths(export_dir)),
        launch_script_path=draft.launch_script_path,
        config_path=draft.config_path,
        extras=draft.extras,
    )


def finalize_vllm_export(
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
        target=VLLM_TARGET.name,
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
        target=VLLM_TARGET.name,
        llama_cpp_tag=None,
        artifacts=artifacts,
        ollama_name=None,
        ollama_version_str=None,
        smoke_first_line=smoke_output_first_line,
        adapter_name=adapter_name,
        adapter_mix=adapter_mix,
    )
    return manifest_path


def _stage_modules(
    *,
    store: StorePath,
    adapters_dir: Path,
    adapter_name: str | None,
    adapter_path_override: Path | None,
    declared_adapter_names: tuple[str, ...] | None,
) -> list[LoraModule]:
    modules = _resolve_modules(
        store=store,
        adapter_name=adapter_name,
        adapter_path_override=adapter_path_override,
        declared_adapter_names=declared_adapter_names,
    )
    staged: list[LoraModule] = []
    for module in modules:
        target_dir = adapters_dir / module.name
        shutil.copytree(module.path, target_dir)
        staged.append(LoraModule(module.name, target_dir, module.adapter_version))
    return staged


def _resolve_modules(
    *,
    store: StorePath,
    adapter_name: str | None,
    adapter_path_override: Path | None,
    declared_adapter_names: tuple[str, ...] | None,
) -> list[LoraModule]:
    if adapter_path_override is not None:
        if not adapter_path_override.exists():
            raise ExportError(f"adapter_path_override {adapter_path_override} does not exist")
        return [
            LoraModule(
                name=_MIXED_MODULE_NAME,
                path=adapter_path_override,
                adapter_version=_version_from_dir_name(adapter_path_override),
            )
        ]

    if adapter_name is not None:
        path = store.resolve_current_adapter_for(adapter_name)
        pointer = store.adapter_current_pointer_for(adapter_name)
        if path is None or not path.exists():
            raise ExportError(
                f"no current adapter under {pointer}; run `dlm train` before exporting."
            )
        return [
            LoraModule(
                name=adapter_name,
                path=path,
                adapter_version=_version_from_dir_name(path),
            )
        ]

    if declared_adapter_names:
        modules: list[LoraModule] = []
        for name in declared_adapter_names:
            path = store.resolve_current_adapter_for(name)
            pointer = store.adapter_current_pointer_for(name)
            if path is None or not path.exists():
                raise ExportError(
                    f"no current adapter under {pointer}; run `dlm train` before exporting."
                )
            modules.append(
                LoraModule(name=name, path=path, adapter_version=_version_from_dir_name(path))
            )
        return modules

    path = store.resolve_current_adapter()
    pointer = store.adapter_current_pointer
    if path is None or not path.exists():
        raise ExportError(f"no current adapter under {pointer}; run `dlm train` before exporting.")
    return [
        LoraModule(
            name=_DEFAULT_MODULE_NAME,
            path=path,
            adapter_version=_version_from_dir_name(path),
        )
    ]


def _version_from_dir_name(path: Path) -> int:
    stem = path.name
    if not stem.startswith("v") or not stem[1:].isdigit():
        return 1
    return int(stem[1:])


def _artifact_paths(export_dir: Path) -> list[Path]:
    artifacts: list[Path] = []
    for path in sorted(export_dir.rglob("*")):
        if path.is_file() and path.name != "export_manifest.json":
            artifacts.append(path)
    return artifacts


def _build_command(prepared: TargetResult, *, use_script_dir: bool) -> list[str]:
    model = _require_prepared_str(prepared, "model")
    revision = _require_prepared_str(prepared, "revision")
    served_model_name = _require_prepared_str(prepared, "served_model_name")
    modules = _require_module_specs(prepared)
    context_length = _optional_prepared_int(prepared, "context_length")

    command = [
        "vllm",
        "serve",
        model,
        "--revision",
        revision,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "auto",
        "--served-model-name",
        served_model_name,
    ]
    if context_length is not None:
        command.extend(["--max-model-len", str(context_length)])
    if modules:
        command.extend(["--enable-lora", "--lora-modules"])
        for module in modules:
            path = (
                f"$SCRIPT_DIR/{_ADAPTERS_DIRNAME}/{module.name}"
                if use_script_dir
                else str(module.path)
            )
            command.append(f"{module.name}={path}")
    return command


def _render_config(prepared: TargetResult) -> str:
    modules = _require_module_specs(prepared)
    payload = {
        "target": VLLM_TARGET.name,
        "model": _require_prepared_str(prepared, "model"),
        "revision": _require_prepared_str(prepared, "revision"),
        "served_model_name": _require_prepared_str(prepared, "served_model_name"),
        "dtype": "auto",
        "host": "127.0.0.1",
        "port": 8000,
        "max_model_len": _optional_prepared_int(prepared, "context_length"),
        "environment": _runtime_env(prepared),
        "lora_modules": [
            {
                "name": module.name,
                "path": f"{_ADAPTERS_DIRNAME}/{module.name}",
                "adapter_version": module.adapter_version,
            }
            for module in modules
        ],
    }
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _render_launch_script(command: list[str], runtime_env: dict[str, str]) -> str:
    rendered = " ".join(_quote_script_arg(arg) for arg in command)
    env_lines = "".join(
        f"export {name}={shlex.quote(value)}\n" for name, value in runtime_env.items()
    )
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
        f"{env_lines}"
        f'exec {rendered} "$@"\n'
    )


def _quote_script_arg(arg: str) -> str:
    if arg.startswith("$SCRIPT_DIR/"):
        return f'"{arg}"'
    if "=$SCRIPT_DIR/" in arg:
        name, value = arg.split("=", 1)
        return f'{shlex.quote(name)}="{value}"'
    return shlex.quote(arg)


def _default_runtime_env() -> dict[str, str]:
    if not _is_darwin_arm64():
        return {}
    # vllm-metal's MLX KV path is the documented low-risk mode on
    # Apple Silicon. The paged-attention path is still experimental
    # and `auto` can expand to a 0.9 memory fraction there, which
    # proved aggressive enough to destabilize local smoke runs.
    return {
        "VLLM_METAL_USE_PAGED_ATTENTION": "0",
        "VLLM_METAL_MEMORY_FRACTION": "auto",
    }


def _runtime_env(prepared: TargetResult) -> dict[str, str]:
    value = prepared.extras.get("runtime_env")
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ExportError("vllm prepared target extra 'runtime_env' must be dict[str, str]")
    return dict(value)


def _is_darwin_arm64() -> bool:
    return _sys_platform() == "darwin" and _machine() == "arm64"


def _sys_platform() -> str:
    return sys.platform


def _machine() -> str:
    return platform.machine()


def _require_prepared_str(prepared: TargetResult, key: str) -> str:
    value = prepared.extras.get(key)
    if not isinstance(value, str) or not value:
        raise ExportError(f"vllm prepared target missing string extra {key!r}")
    return value


def _require_prepared_int(prepared: TargetResult, key: str) -> int:
    value = prepared.extras.get(key)
    if not isinstance(value, int):
        raise ExportError(f"vllm prepared target missing int extra {key!r}")
    return value


def _optional_prepared_int(prepared: TargetResult, key: str) -> int | None:
    value = prepared.extras.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ExportError(f"vllm prepared target extra {key!r} must be an int")
    return value


def _require_module_specs(prepared: TargetResult) -> tuple[LoraModule, ...]:
    value = prepared.extras.get("module_specs")
    if not isinstance(value, tuple) or not all(isinstance(item, LoraModule) for item in value):
        raise ExportError("vllm prepared target missing LoraModule tuple extra 'module_specs'")
    return value


VLLM_TARGET = VllmTarget()
assert isinstance(VLLM_TARGET, ExportTarget)
