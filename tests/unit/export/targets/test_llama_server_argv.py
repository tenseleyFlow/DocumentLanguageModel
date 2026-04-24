"""llama-server launch artifact generation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from dlm.base_models import BASE_MODELS
from dlm.export.dispatch import DispatchResult
from dlm.export.errors import ExportError
from dlm.export.manifest import ExportManifest, load_export_manifest
from dlm.export.targets.base import TargetResult
from dlm.export.targets.llama_server import (
    LLAMA_SERVER_TARGET,
    _find_artifact,
    _optional_int_extra,
    _optional_path_extra,
    _optional_prepared_path,
    _read_chat_template,
    _require_path_extra,
    _require_prepared_int,
    _require_prepared_path,
    _require_spec_extra,
    _script_dir_arg,
    prepare_llama_server_export,
)


def _vendor_tree(tmp_path: Path) -> Path:
    vendor = tmp_path / "vendor" / "llama.cpp"
    (vendor / "build" / "bin").mkdir(parents=True)
    server = vendor / "build" / "bin" / "llama-server"
    server.write_text("#!/bin/sh\n")
    server.chmod(0o755)
    (vendor / "convert_hf_to_gguf.py").write_text("# mock\n")
    (vendor / "convert_lora_to_gguf.py").write_text("# mock\n")
    return vendor


def _adapter_dir(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}"})
    )
    return adapter_dir


def _seed_manifest(export_dir: Path) -> None:
    manifest = ExportManifest(
        target="llama-server",
        quant="Q4_K_M",
        created_at=datetime(2026, 4, 23, 12, 0, 0),
        created_by="dlm-test",
        base_model_hf_id="org/base",
        base_model_revision="a" * 40,
        adapter_version=1,
        artifacts=[],
    )
    (export_dir / "export_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


class TestPrepareLlamaServerExport:
    def test_writes_template_and_launch_script_for_unmerged_export(self, tmp_path: Path) -> None:
        export_dir = tmp_path / "exports" / "Q4_K_M"
        export_dir.mkdir(parents=True)
        manifest_path = export_dir / "export_manifest.json"
        _seed_manifest(export_dir)
        base = export_dir / "base.Q4_K_M.gguf"
        adapter_gguf = export_dir / "adapter.gguf"
        base.write_bytes(b"base")
        adapter_gguf.write_bytes(b"adapter")
        adapter_dir = _adapter_dir(tmp_path)
        vendor = _vendor_tree(tmp_path)

        prepared = prepare_llama_server_export(
            export_dir=export_dir,
            manifest_path=manifest_path,
            artifacts=[base, adapter_gguf],
            adapter_dir=adapter_dir,
            spec=BASE_MODELS["smollm2-135m"],
            training_sequence_len=4096,
            vendor_override=vendor,
        )

        assert prepared.name == "llama-server"
        assert prepared.launch_script_path is not None
        assert prepared.launch_script_path.exists()
        assert prepared.config_path is not None
        assert prepared.config_path.exists()

        script = prepared.launch_script_path.read_text(encoding="utf-8")
        assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert 'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"' in script
        assert '--model "$SCRIPT_DIR/base.Q4_K_M.gguf"' in script
        assert '--chat-template-file "$SCRIPT_DIR/chat-template.jinja"' in script
        assert '--lora "$SCRIPT_DIR/adapter.gguf"' in script
        assert "--ctx-size 4096" in script
        assert "--host 127.0.0.1 --port 8000" in script
        assert "llama-server" in script

        manifest = load_export_manifest(export_dir)
        assert [artifact.path for artifact in manifest.artifacts] == [
            "chat-template.jinja",
            "llama-server_launch.sh",
        ]

    def test_merged_export_omits_lora_flag(self, tmp_path: Path) -> None:
        export_dir = tmp_path / "exports" / "Q4_K_M"
        export_dir.mkdir(parents=True)
        manifest_path = export_dir / "export_manifest.json"
        _seed_manifest(export_dir)
        base = export_dir / "base.Q4_K_M.gguf"
        base.write_bytes(b"base")
        adapter_dir = _adapter_dir(tmp_path)
        vendor = _vendor_tree(tmp_path)

        prepared = prepare_llama_server_export(
            export_dir=export_dir,
            manifest_path=manifest_path,
            artifacts=[base],
            adapter_dir=adapter_dir,
            spec=BASE_MODELS["smollm2-135m"],
            training_sequence_len=512,
            vendor_override=vendor,
        )

        assert prepared.launch_script_path is not None
        script = prepared.launch_script_path.read_text(encoding="utf-8")
        assert "--lora " not in script
        assert "--ctx-size 512" in script


class TestLlamaServerHelpers:
    def test_read_chat_template_rejects_invalid_json(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "tokenizer_config.json").write_text("not json {{{", encoding="utf-8")

        with pytest.raises(ExportError, match="cannot load chat template"):
            _read_chat_template(adapter_dir)

    def test_read_chat_template_rejects_blank_template(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "   "}),
            encoding="utf-8",
        )

        with pytest.raises(ExportError, match="has no non-empty chat_template"):
            _read_chat_template(adapter_dir)

    def test_find_artifact_missing_prefix_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="missing export artifact with prefix"):
            _find_artifact([tmp_path / "adapter.gguf"], prefix="base.")

    def test_script_dir_arg_requires_path(self) -> None:
        with pytest.raises(ExportError, match="missing a required path"):
            _script_dir_arg(None)

    def test_dispatch_extra_validators_raise_on_wrong_types(self, tmp_path: Path) -> None:
        ctx = DispatchResult(
            export_dir=tmp_path,
            manifest_path=tmp_path / "export_manifest.json",
            artifacts=[],
            banner_lines=[],
            extras={
                "adapter_dir": "bad",
                "training_sequence_len": "bad",
                "spec": "bad",
                "vendor_override": "bad",
            },
        )

        with pytest.raises(ExportError, match="missing Path extra 'adapter_dir'"):
            _require_path_extra(ctx, "adapter_dir")
        with pytest.raises(ExportError, match="must be an int"):
            _optional_int_extra(ctx, "training_sequence_len")
        with pytest.raises(ExportError, match="missing BaseModelSpec extra 'spec'"):
            _require_spec_extra(ctx, "spec")
        with pytest.raises(ExportError, match="must be a Path"):
            _optional_path_extra(ctx, "vendor_override")

        empty_ctx = DispatchResult(
            export_dir=tmp_path,
            manifest_path=tmp_path / "export_manifest.json",
            artifacts=[],
            banner_lines=[],
            extras={},
        )
        assert _optional_path_extra(empty_ctx, "vendor_override") is None
        assert _optional_int_extra(empty_ctx, "training_sequence_len") is None

    def test_prepared_extra_validators_raise_on_wrong_types(self, tmp_path: Path) -> None:
        prepared = TargetResult(
            name="llama-server",
            export_dir=tmp_path,
            manifest_path=tmp_path / "export_manifest.json",
            config_path=tmp_path / "chat-template.jinja",
            extras={
                "model_path": tmp_path / "base.gguf",
                "adapter_gguf_path": "bad",
                "context_length": 512,
            },
        )

        assert _require_prepared_path(prepared, "model_path") == tmp_path / "base.gguf"
        with pytest.raises(ExportError, match="must be a Path"):
            _optional_prepared_path(prepared, "adapter_gguf_path")
        with pytest.raises(ExportError, match="must be a Path"):
            LLAMA_SERVER_TARGET.launch_command(prepared)

        bad_int = TargetResult(
            name="llama-server",
            export_dir=tmp_path,
            manifest_path=tmp_path / "export_manifest.json",
            config_path=tmp_path / "chat-template.jinja",
            extras={
                "model_path": tmp_path / "base.gguf",
                "context_length": "bad",
            },
        )
        with pytest.raises(ExportError, match="missing int extra 'context_length'"):
            _require_prepared_int(bad_int, "context_length")

    def test_smoke_failure_from_runtime_command_is_reported(self, tmp_path: Path) -> None:
        prepared = TargetResult(
            name="llama-server",
            export_dir=tmp_path,
            manifest_path=tmp_path / "export_manifest.json",
            extras={"model_path": "bad"},
            config_path=tmp_path / "chat-template.jinja",
        )

        result = LLAMA_SERVER_TARGET.smoke_test(prepared)

        assert result.attempted is True
        assert result.ok is False
        assert "missing Path extra 'model_path'" in result.detail
