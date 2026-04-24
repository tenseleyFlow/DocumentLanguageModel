"""vLLM launch artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

from dlm.base_models import BASE_MODELS
from dlm.export.manifest import load_export_manifest
from dlm.export.targets.vllm import (
    VLLM_CONFIG_FILENAME,
    VLLM_TARGET,
    finalize_vllm_export,
    prepare_vllm_export,
)
from dlm.store.manifest import Manifest, load_manifest, save_manifest
from dlm.store.paths import for_dlm

_SPEC = BASE_MODELS["smollm2-135m"]


def _write_adapter(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{messages}}", "vocab_size": 32000}),
        encoding="utf-8",
    )


def _setup_flat_store(tmp_path: Path) -> object:
    store = for_dlm("01VLLMTEST", home=tmp_path)
    store.ensure_layout()
    save_manifest(store.manifest, Manifest(dlm_id="01VLLMTEST", base_model=_SPEC.key))
    adapter = store.adapter_version(3)
    _write_adapter(adapter)
    store.set_current_adapter(adapter)
    return store


def _setup_named_store(tmp_path: Path) -> object:
    store = for_dlm("01VLLMMULTI", home=tmp_path)
    store.ensure_layout()
    save_manifest(store.manifest, Manifest(dlm_id="01VLLMMULTI", base_model=_SPEC.key))
    knowledge = store.adapter_version_for("knowledge", 2)
    tone = store.adapter_version_for("tone", 4)
    _write_adapter(knowledge)
    _write_adapter(tone)
    store.set_current_adapter_for("knowledge", knowledge)
    store.set_current_adapter_for("tone", tone)
    return store


class TestPrepareVllmExport:
    def test_flat_export_writes_config_manifest_and_launch_script(self, tmp_path: Path) -> None:
        store = _setup_flat_store(tmp_path)

        prepared = prepare_vllm_export(
            store=store,
            spec=_SPEC,
            served_model_name="dlm-flat",
            training_sequence_len=2048,
            adapter_name=None,
            adapter_path_override=None,
            declared_adapter_names=None,
        )
        manifest_path = finalize_vllm_export(
            store=store,
            spec=_SPEC,
            prepared=prepared,
            smoke_output_first_line="hello from vllm",
            adapter_name=None,
            adapter_mix=None,
        )

        assert prepared.launch_script_path is not None
        script = prepared.launch_script_path.read_text(encoding="utf-8")
        assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert "vllm serve" in script
        assert _SPEC.hf_id in script
        assert "--revision" in script
        assert "--served-model-name dlm-flat" in script
        assert "--max-model-len 2048" in script
        assert 'adapter="$SCRIPT_DIR/adapters/adapter"' in script

        config = json.loads(
            (prepared.export_dir / VLLM_CONFIG_FILENAME).read_text(encoding="utf-8")
        )
        assert config["target"] == "vllm"
        assert config["model"] == _SPEC.hf_id
        assert config["served_model_name"] == "dlm-flat"
        assert config["max_model_len"] == 2048
        assert config["lora_modules"] == [
            {"adapter_version": 3, "name": "adapter", "path": "adapters/adapter"}
        ]

        export_manifest = load_export_manifest(prepared.export_dir)
        assert manifest_path == prepared.manifest_path
        assert export_manifest.target == "vllm"
        assert export_manifest.quant == "hf"
        assert export_manifest.adapter_version == 3
        assert any(artifact.path == "vllm_launch.sh" for artifact in export_manifest.artifacts)
        assert any(artifact.path == "vllm_config.json" for artifact in export_manifest.artifacts)
        assert any(
            artifact.path == "adapters/adapter/adapter_model.safetensors"
            for artifact in export_manifest.artifacts
        )

        store_manifest = load_manifest(store.manifest)
        assert store_manifest.exports[-1].target == "vllm"
        assert store_manifest.exports[-1].quant == "hf"
        assert store_manifest.exports[-1].smoke_output_first_line == "hello from vllm"

    def test_multi_adapter_export_includes_all_named_modules(self, tmp_path: Path) -> None:
        store = _setup_named_store(tmp_path)

        prepared = prepare_vllm_export(
            store=store,
            spec=_SPEC,
            served_model_name="dlm-multi",
            training_sequence_len=4096,
            adapter_name=None,
            adapter_path_override=None,
            declared_adapter_names=("knowledge", "tone"),
        )

        script = prepared.launch_script_path.read_text(encoding="utf-8")
        assert "--max-model-len 4096" in script
        assert 'knowledge="$SCRIPT_DIR/adapters/knowledge"' in script
        assert 'tone="$SCRIPT_DIR/adapters/tone"' in script

        config = json.loads(
            (prepared.export_dir / VLLM_CONFIG_FILENAME).read_text(encoding="utf-8")
        )
        assert config["max_model_len"] == 4096
        assert config["lora_modules"] == [
            {"adapter_version": 2, "name": "knowledge", "path": "adapters/knowledge"},
            {"adapter_version": 4, "name": "tone", "path": "adapters/tone"},
        ]

    def test_adapter_mix_override_stages_one_mixed_module(self, tmp_path: Path) -> None:
        store = _setup_named_store(tmp_path)
        mixed = tmp_path / "mixed"
        _write_adapter(mixed)

        prepared = prepare_vllm_export(
            store=store,
            spec=_SPEC,
            served_model_name="dlm-mixed",
            training_sequence_len=1024,
            adapter_name=None,
            adapter_path_override=mixed,
            declared_adapter_names=("knowledge", "tone"),
        )

        script = prepared.launch_script_path.read_text(encoding="utf-8")
        assert "--served-model-name dlm-mixed" in script
        assert "--max-model-len 1024" in script
        assert 'mixed="$SCRIPT_DIR/adapters/mixed"' in script
        assert 'knowledge="$SCRIPT_DIR/adapters/knowledge"' not in script
        assert 'tone="$SCRIPT_DIR/adapters/tone"' not in script

        config = json.loads(
            (prepared.export_dir / VLLM_CONFIG_FILENAME).read_text(encoding="utf-8")
        )
        assert config["lora_modules"] == [
            {"adapter_version": 1, "name": "mixed", "path": "adapters/mixed"}
        ]

    def test_apple_silicon_export_records_conservative_runtime_env(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        store = _setup_flat_store(tmp_path)
        monkeypatch.setattr("dlm.export.targets.vllm._sys_platform", lambda: "darwin")
        monkeypatch.setattr("dlm.export.targets.vllm._machine", lambda: "arm64")

        prepared = prepare_vllm_export(
            store=store,
            spec=_SPEC,
            served_model_name="dlm-flat",
            training_sequence_len=2048,
            adapter_name=None,
            adapter_path_override=None,
            declared_adapter_names=None,
        )

        script = prepared.launch_script_path.read_text(encoding="utf-8")
        assert "export VLLM_METAL_USE_PAGED_ATTENTION=0" in script
        assert "export VLLM_METAL_MEMORY_FRACTION=auto" in script

        config = json.loads(
            (prepared.export_dir / VLLM_CONFIG_FILENAME).read_text(encoding="utf-8")
        )
        assert config["environment"] == {
            "VLLM_METAL_MEMORY_FRACTION": "auto",
            "VLLM_METAL_USE_PAGED_ATTENTION": "0",
        }


class TestVllmSmoke:
    def test_smoke_uses_absolute_runtime_paths(self, tmp_path: Path, monkeypatch: object) -> None:
        store = _setup_named_store(tmp_path)
        monkeypatch.setattr("dlm.export.targets.vllm._sys_platform", lambda: "darwin")
        monkeypatch.setattr("dlm.export.targets.vllm._machine", lambda: "arm64")
        prepared = prepare_vllm_export(
            store=store,
            spec=_SPEC,
            served_model_name="dlm-multi",
            training_sequence_len=2048,
            adapter_name=None,
            adapter_path_override=None,
            declared_adapter_names=("knowledge", "tone"),
        )
        seen: list[tuple[list[str], object]] = []

        def _fake_smoke(argv: list[str], **kwargs: object) -> str:
            seen.append((list(argv), kwargs.get("env")))
            return "vllm replied"

        monkeypatch.setattr("dlm.export.targets.vllm.smoke_openai_compat_server", _fake_smoke)

        result = VLLM_TARGET.smoke_test(prepared)

        assert result.attempted is True
        assert result.ok is True
        assert result.detail == "vllm replied"
        argv, env = seen[0]
        assert argv[:2] == ["vllm", "serve"]
        assert "$SCRIPT_DIR" not in " ".join(argv)
        assert _SPEC.hf_id in argv
        assert "--max-model-len" in argv
        assert "2048" in argv
        assert env == {
            "VLLM_METAL_MEMORY_FRACTION": "auto",
            "VLLM_METAL_USE_PAGED_ATTENTION": "0",
        }
        assert f"knowledge={prepared.export_dir / 'adapters' / 'knowledge'}" in argv
        assert f"tone={prepared.export_dir / 'adapters' / 'tone'}" in argv
