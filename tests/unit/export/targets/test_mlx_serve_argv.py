"""MLX serve launch artifact generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.base_models import BASE_MODELS
from dlm.export.errors import ExportError, TargetSmokeError
from dlm.export.manifest import load_export_manifest
from dlm.export.targets.base import TargetResult
from dlm.export.targets.mlx_serve import (
    LAUNCH_SCRIPT_FILENAME,
    MLX_SERVE_TARGET,
    _quote_script_arg,
    _require_prepared_int,
    _require_prepared_path,
    _require_prepared_str,
    _version_from_dir_name,
    finalize_mlx_serve_export,
    prepare_mlx_serve_export,
)
from dlm.store.manifest import Manifest, load_manifest, save_manifest
from dlm.store.paths import for_dlm

_SPEC = BASE_MODELS["smollm2-135m"]


def _write_adapter(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")


def _fake_stage_mlx(src: Path, dst: Path, *, base_hf_id: str) -> Path:
    assert src.exists()
    assert base_hf_id == _SPEC.hf_id
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "adapter_config.json").write_text("{}", encoding="utf-8")
    (dst / "adapters.safetensors").write_bytes(b"mlx-adapter")
    return dst


def _setup_flat_store(tmp_path: Path) -> object:
    store = for_dlm("01MLXTEST", home=tmp_path)
    store.ensure_layout()
    save_manifest(store.manifest, Manifest(dlm_id="01MLXTEST", base_model=_SPEC.key))
    adapter = store.adapter_version(3)
    _write_adapter(adapter)
    store.set_current_adapter(adapter)
    return store


def _setup_named_store(tmp_path: Path) -> object:
    store = for_dlm("01MLXMULTI", home=tmp_path)
    store.ensure_layout()
    save_manifest(store.manifest, Manifest(dlm_id="01MLXMULTI", base_model=_SPEC.key))
    knowledge = store.adapter_version_for("knowledge", 2)
    tone = store.adapter_version_for("tone", 4)
    _write_adapter(knowledge)
    _write_adapter(tone)
    store.set_current_adapter_for("knowledge", knowledge)
    store.set_current_adapter_for("tone", tone)
    return store


class TestPrepareMlxServeExport:
    def test_prepare_method_is_not_used_directly(self) -> None:
        with pytest.raises(NotImplementedError, match="prepare_mlx_serve_export"):
            MLX_SERVE_TARGET.prepare(object())

    def test_prepare_writes_launch_script_and_manifest(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        store = _setup_flat_store(tmp_path)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.is_apple_silicon", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.mlx_available", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.stage_mlx_adapter_dir", _fake_stage_mlx)

        prepared = prepare_mlx_serve_export(
            store=store,
            spec=_SPEC,
            adapter_name=None,
            adapter_path_override=None,
            declared_adapter_names=None,
        )
        manifest_path = finalize_mlx_serve_export(
            store=store,
            spec=_SPEC,
            prepared=prepared,
            smoke_output_first_line="hello from mlx",
            adapter_name=None,
            adapter_mix=None,
        )

        assert prepared.launch_script_path is not None
        assert prepared.launch_script_path.name == LAUNCH_SCRIPT_FILENAME
        script = prepared.launch_script_path.read_text(encoding="utf-8")
        assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert "python -m mlx_lm.server" in script
        assert f"--model {_SPEC.hf_id}" in script
        assert '--adapter-path "$SCRIPT_DIR/adapter"' in script

        export_manifest = load_export_manifest(prepared.export_dir)
        assert manifest_path == prepared.manifest_path
        assert export_manifest.target == "mlx-serve"
        assert export_manifest.quant == "hf"
        assert export_manifest.adapter_version == 3
        assert any(artifact.path == "mlx_serve_launch.sh" for artifact in export_manifest.artifacts)
        assert any(
            artifact.path == "adapter/adapters.safetensors"
            for artifact in export_manifest.artifacts
        )

        store_manifest = load_manifest(store.manifest)
        assert store_manifest.exports[-1].target == "mlx-serve"
        assert store_manifest.exports[-1].quant == "hf"
        assert store_manifest.exports[-1].smoke_output_first_line == "hello from mlx"

    def test_prepare_replaces_stale_staged_adapter_dir(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        store = _setup_flat_store(tmp_path)
        export_dir = store.exports / "mlx-serve"
        stale_dir = export_dir / "adapter"
        stale_dir.mkdir(parents=True)
        (stale_dir / "stale.txt").write_text("stale", encoding="utf-8")
        monkeypatch.setattr("dlm.export.targets.mlx_serve.is_apple_silicon", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.mlx_available", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.stage_mlx_adapter_dir", _fake_stage_mlx)

        prepared = prepare_mlx_serve_export(
            store=store,
            spec=_SPEC,
            adapter_name=None,
            adapter_path_override=None,
            declared_adapter_names=None,
        )

        assert prepared.launch_script_path is not None
        assert not (prepared.export_dir / "adapter" / "stale.txt").exists()
        assert (prepared.export_dir / "adapter" / "adapters.safetensors").exists()

    def test_multi_adapter_export_requires_explicit_selection(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        store = _setup_named_store(tmp_path)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.is_apple_silicon", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.mlx_available", lambda: True)

        with pytest.raises(ExportError, match="one adapter at a time"):
            prepare_mlx_serve_export(
                store=store,
                spec=_SPEC,
                adapter_name=None,
                adapter_path_override=None,
                declared_adapter_names=("knowledge", "tone"),
            )

    def test_refuses_without_apple_silicon_runtime(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        store = _setup_flat_store(tmp_path)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.is_apple_silicon", lambda: False)

        with pytest.raises(ExportError, match="Apple Silicon"):
            prepare_mlx_serve_export(
                store=store,
                spec=_SPEC,
                adapter_name=None,
                adapter_path_override=None,
                declared_adapter_names=None,
            )

    def test_refuses_without_mlx_extra(self, tmp_path: Path, monkeypatch: object) -> None:
        store = _setup_flat_store(tmp_path)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.is_apple_silicon", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.mlx_available", lambda: False)

        with pytest.raises(ExportError, match="mlx extra"):
            prepare_mlx_serve_export(
                store=store,
                spec=_SPEC,
                adapter_name=None,
                adapter_path_override=None,
                declared_adapter_names=None,
            )

    def test_missing_named_adapter_raises(self, tmp_path: Path, monkeypatch: object) -> None:
        store = _setup_named_store(tmp_path)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.is_apple_silicon", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.mlx_available", lambda: True)

        with pytest.raises(ExportError, match="no current adapter under"):
            prepare_mlx_serve_export(
                store=store,
                spec=_SPEC,
                adapter_name="missing",
                adapter_path_override=None,
                declared_adapter_names=None,
            )

    def test_named_adapter_export_uses_named_dir(self, tmp_path: Path, monkeypatch: object) -> None:
        store = _setup_named_store(tmp_path)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.is_apple_silicon", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.mlx_available", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.stage_mlx_adapter_dir", _fake_stage_mlx)

        prepared = prepare_mlx_serve_export(
            store=store,
            spec=_SPEC,
            adapter_name="knowledge",
            adapter_path_override=None,
            declared_adapter_names=None,
        )

        assert str(prepared.extras["adapter_dir"]).endswith("knowledge")
        assert prepared.extras["adapter_version"] == 2

    def test_missing_adapter_override_raises(self, tmp_path: Path, monkeypatch: object) -> None:
        store = _setup_flat_store(tmp_path)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.is_apple_silicon", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.mlx_available", lambda: True)

        with pytest.raises(ExportError, match="adapter_path_override .* does not exist"):
            prepare_mlx_serve_export(
                store=store,
                spec=_SPEC,
                adapter_name=None,
                adapter_path_override=tmp_path / "missing",
                declared_adapter_names=None,
            )

    def test_existing_adapter_override_uses_mixed_dir(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        store = _setup_flat_store(tmp_path)
        override = tmp_path / "custom-adapter"
        _write_adapter(override)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.is_apple_silicon", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.mlx_available", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.stage_mlx_adapter_dir", _fake_stage_mlx)

        prepared = prepare_mlx_serve_export(
            store=store,
            spec=_SPEC,
            adapter_name=None,
            adapter_path_override=override,
            declared_adapter_names=None,
        )

        assert str(prepared.extras["adapter_dir"]).endswith("mixed")
        assert prepared.extras["adapter_version"] == 1

    def test_missing_default_adapter_raises(self, tmp_path: Path, monkeypatch: object) -> None:
        store = for_dlm("01EMPTYMLX", home=tmp_path)
        store.ensure_layout()
        save_manifest(store.manifest, Manifest(dlm_id="01EMPTYMLX", base_model=_SPEC.key))
        monkeypatch.setattr("dlm.export.targets.mlx_serve.is_apple_silicon", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.mlx_available", lambda: True)

        with pytest.raises(ExportError, match="no current adapter under"):
            prepare_mlx_serve_export(
                store=store,
                spec=_SPEC,
                adapter_name=None,
                adapter_path_override=None,
                declared_adapter_names=None,
            )


class TestMlxServeSmoke:
    def test_smoke_uses_absolute_runtime_paths(self, tmp_path: Path, monkeypatch: object) -> None:
        store = _setup_flat_store(tmp_path)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.is_apple_silicon", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.mlx_available", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.stage_mlx_adapter_dir", _fake_stage_mlx)
        prepared = prepare_mlx_serve_export(
            store=store,
            spec=_SPEC,
            adapter_name=None,
            adapter_path_override=None,
            declared_adapter_names=None,
        )
        seen: list[list[str]] = []

        def _fake_smoke(argv: list[str], **_: object) -> str:
            seen.append(list(argv))
            return "mlx replied"

        monkeypatch.setattr("dlm.export.targets.mlx_serve.smoke_openai_compat_server", _fake_smoke)

        result = MLX_SERVE_TARGET.smoke_test(prepared)

        assert result.attempted is True
        assert result.ok is True
        assert result.detail == "mlx replied"
        argv = seen[0]
        assert argv[:3] == ["python", "-m", "mlx_lm.server"]
        assert "$SCRIPT_DIR" not in " ".join(argv)
        assert _SPEC.hf_id in argv
        assert str(prepared.export_dir / "adapter") in argv

    def test_smoke_failure_returns_failed_result(self, tmp_path: Path, monkeypatch: object) -> None:
        store = _setup_flat_store(tmp_path)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.is_apple_silicon", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.mlx_available", lambda: True)
        monkeypatch.setattr("dlm.export.targets.mlx_serve.stage_mlx_adapter_dir", _fake_stage_mlx)
        prepared = prepare_mlx_serve_export(
            store=store,
            spec=_SPEC,
            adapter_name=None,
            adapter_path_override=None,
            declared_adapter_names=None,
        )

        def _fake_smoke(argv: list[str], **_: object) -> str:
            _ = argv
            raise TargetSmokeError("boom")

        monkeypatch.setattr("dlm.export.targets.mlx_serve.smoke_openai_compat_server", _fake_smoke)

        result = MLX_SERVE_TARGET.smoke_test(prepared)

        assert result.attempted is True
        assert result.ok is False
        assert result.detail == "boom"


class TestMlxServeHelpers:
    def test_quote_script_arg_and_int_validation(self) -> None:
        assert _quote_script_arg("$SCRIPT_DIR/adapter") == '"$SCRIPT_DIR/adapter"'
        assert _quote_script_arg("plain value") == "'plain value'"

        prepared = TargetResult(
            name="mlx-serve",
            export_dir=Path("/tmp/export"),
            manifest_path=Path("/tmp/export/export_manifest.json"),
            extras={"adapter_version": "bad"},
        )
        with pytest.raises(ExportError, match="missing int extra"):
            _require_prepared_int(prepared, "adapter_version")

    def test_string_and_path_validation(self) -> None:
        prepared = TargetResult(
            name="mlx-serve",
            export_dir=Path("/tmp/export"),
            manifest_path=Path("/tmp/export/export_manifest.json"),
            extras={"model": "", "adapter_dir": "bad"},
        )

        with pytest.raises(ExportError, match="missing string extra"):
            _require_prepared_str(prepared, "model")
        with pytest.raises(ExportError, match="missing Path extra"):
            _require_prepared_path(prepared, "adapter_dir")

    def test_version_from_dir_name_defaults_for_non_version_dirs(self) -> None:
        assert _version_from_dir_name(Path("custom-adapter")) == 1
