"""llama.cpp-server target helpers."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from dlm.export.dispatch import DispatchResult
from dlm.export.errors import ExportError
from dlm.export.manifest import build_artifact, load_export_manifest, save_export_manifest
from dlm.export.ollama.modelfile_shared import resolve_num_ctx
from dlm.export.targets.base import ExportTarget, SmokeResult, TargetResult
from dlm.export.vendoring import llama_server_bin
from dlm.io.atomic import write_text

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec


CHAT_TEMPLATE_FILENAME = "chat-template.jinja"
LAUNCH_SCRIPT_FILENAME = "llama-server_launch.sh"


class LlamaServerTarget:
    """Registered export target for llama.cpp-server launch artifacts."""

    name = "llama-server"

    def prepare(self, ctx: DispatchResult) -> TargetResult:
        model_path = _require_path_extra(ctx, "model_path")
        adapter_dir = _require_path_extra(ctx, "adapter_dir")
        context_length = _require_int_extra(ctx, "context_length")
        adapter_gguf_path = _optional_path_extra(ctx, "adapter_gguf_path")
        vendor_override = _optional_path_extra(ctx, "vendor_override")

        template_path = ctx.export_dir / CHAT_TEMPLATE_FILENAME
        write_text(template_path, _read_chat_template(adapter_dir))

        launch_script_path = ctx.export_dir / LAUNCH_SCRIPT_FILENAME
        prepared = TargetResult(
            name=self.name,
            export_dir=ctx.export_dir,
            manifest_path=ctx.manifest_path,
            artifacts=(*ctx.artifacts, template_path, launch_script_path),
            launch_script_path=launch_script_path,
            config_path=template_path,
            extras={
                "model_path": model_path,
                "adapter_gguf_path": adapter_gguf_path,
                "context_length": context_length,
                "vendor_override": vendor_override,
            },
        )
        write_text(launch_script_path, _render_launch_script(self.launch_command(prepared)))
        launch_script_path.chmod(0o755)
        _record_launch_artifacts(ctx.export_dir, template_path, launch_script_path)
        return prepared

    def launch_command(self, prepared: TargetResult) -> list[str]:
        model_path = _require_prepared_path(prepared, "model_path")
        adapter_gguf_path = _optional_prepared_path(prepared, "adapter_gguf_path")
        context_length = _require_prepared_int(prepared, "context_length")
        vendor_override = _optional_prepared_path(prepared, "vendor_override")

        command = [
            str(llama_server_bin(vendor_override)),
            "--model",
            _script_dir_arg(model_path),
            "--api-key",
            "disabled",
            "--ctx-size",
            str(context_length),
            "--chat-template-file",
            _script_dir_arg(prepared.config_path),
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
        if adapter_gguf_path is not None:
            command.extend(["--lora", _script_dir_arg(adapter_gguf_path)])
        return command

    def smoke_test(self, prepared: TargetResult) -> SmokeResult:
        _ = prepared
        return SmokeResult(
            attempted=False,
            ok=True,
            detail="llama-server HTTP smoke lands in a follow-up Sprint 41 slice",
        )


def prepare_llama_server_export(
    *,
    export_dir: Path,
    manifest_path: Path,
    artifacts: list[Path],
    adapter_dir: Path,
    spec: BaseModelSpec,
    training_sequence_len: int | None,
    vendor_override: Path | None = None,
) -> TargetResult:
    """Build launch artifacts for a text GGUF export."""

    model_path = _find_artifact(artifacts, prefix="base.")
    adapter_gguf_path = _find_optional_artifact(artifacts, exact_name="adapter.gguf")
    context_length = resolve_num_ctx(training_sequence_len, spec.context_length)
    ctx = DispatchResult(
        export_dir=export_dir,
        manifest_path=manifest_path,
        artifacts=list(artifacts),
        banner_lines=[],
        extras={
            "model_path": model_path,
            "adapter_dir": adapter_dir,
            "adapter_gguf_path": adapter_gguf_path,
            "context_length": context_length,
            "vendor_override": vendor_override,
        },
    )
    return LLAMA_SERVER_TARGET.prepare(ctx)


def _read_chat_template(adapter_dir: Path) -> str:
    cfg_path = adapter_dir / "tokenizer_config.json"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot load chat template from {cfg_path}: {exc}") from exc
    template = data.get("chat_template")
    if not isinstance(template, str) or not template.strip():
        raise ExportError(f"{cfg_path} has no non-empty chat_template for llama-server export")
    return template.rstrip() + "\n"


def _find_artifact(artifacts: list[Path], *, prefix: str) -> Path:
    for artifact in artifacts:
        if artifact.name.startswith(prefix):
            return artifact
    raise ExportError(f"missing export artifact with prefix {prefix!r}")


def _find_optional_artifact(artifacts: list[Path], *, exact_name: str) -> Path | None:
    for artifact in artifacts:
        if artifact.name == exact_name:
            return artifact
    return None


def _script_dir_arg(path: Path | None) -> str:
    if path is None:
        raise ExportError("llama-server launch artifact missing a required path")
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


def _record_launch_artifacts(
    export_dir: Path, template_path: Path, launch_script_path: Path
) -> None:
    manifest = load_export_manifest(export_dir)
    updated = manifest.model_copy(
        update={
            "artifacts": [
                *manifest.artifacts,
                build_artifact(export_dir, template_path),
                build_artifact(export_dir, launch_script_path),
            ]
        }
    )
    save_export_manifest(export_dir, updated)


def _require_path_extra(ctx: DispatchResult, key: str) -> Path:
    value = ctx.extras.get(key)
    if not isinstance(value, Path):
        raise ExportError(f"llama-server target missing Path extra {key!r}")
    return value


def _optional_path_extra(ctx: DispatchResult, key: str) -> Path | None:
    value = ctx.extras.get(key)
    if value is None:
        return None
    if not isinstance(value, Path):
        raise ExportError(f"llama-server target extra {key!r} must be a Path")
    return value


def _require_int_extra(ctx: DispatchResult, key: str) -> int:
    value = ctx.extras.get(key)
    if not isinstance(value, int):
        raise ExportError(f"llama-server target missing int extra {key!r}")
    return value


def _require_prepared_path(prepared: TargetResult, key: str) -> Path:
    value = prepared.extras.get(key)
    if not isinstance(value, Path):
        raise ExportError(f"llama-server prepared target missing Path extra {key!r}")
    return value


def _optional_prepared_path(prepared: TargetResult, key: str) -> Path | None:
    value = prepared.extras.get(key)
    if value is None:
        return None
    if not isinstance(value, Path):
        raise ExportError(f"llama-server prepared extra {key!r} must be a Path")
    return value


def _require_prepared_int(prepared: TargetResult, key: str) -> int:
    value = prepared.extras.get(key)
    if not isinstance(value, int):
        raise ExportError(f"llama-server prepared target missing int extra {key!r}")
    return value


LLAMA_SERVER_TARGET = LlamaServerTarget()
assert isinstance(LLAMA_SERVER_TARGET, ExportTarget)
