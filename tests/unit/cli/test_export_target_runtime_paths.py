"""CLI coverage for vLLM / MLX runtime-target success and smoke paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from dlm.base_models import BaseModelSpec
from dlm.cli.app import app
from dlm.export.errors import ExportError


def _joined_output(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(text.split())


def _scaffold_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "doc.dlm"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_path / "home"),
            "init",
            str(doc),
            "--base",
            "smollm2-135m",
        ],
    )
    assert result.exit_code == 0, result.output
    return doc


def _spec() -> BaseModelSpec:
    return BaseModelSpec.model_validate(
        {
            "key": "demo-1b",
            "hf_id": "org/demo-1b",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "architecture": "DemoForCausalLM",
            "params": 1_000_000_000,
            "target_modules": ["q_proj", "v_proj"],
            "template": "chatml",
            "gguf_arch": "demo",
            "tokenizer_pre": "demo",
            "license_spdx": "Apache-2.0",
            "license_url": None,
            "requires_acceptance": False,
            "redistributable": True,
            "size_gb_fp16": 2.0,
            "context_length": 4096,
            "recommended_seq_len": 2048,
        }
    )


def _patch_text_export_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dlm.base_models.resolve", lambda *args, **kwargs: _spec())
    monkeypatch.setattr(
        "dlm.base_models.download_spec",
        lambda *args, **kwargs: SimpleNamespace(path=Path("/tmp/base-cache")),
    )
    monkeypatch.setattr(
        "dlm.modality.modality_for",
        lambda spec: SimpleNamespace(accepts_images=False, accepts_audio=False),
    )
    monkeypatch.setattr(
        "dlm.export.gate_fallback.resolve_and_announce",
        lambda store, parsed: SimpleNamespace(entries=None, banner_lines=[]),
    )


class _FakeTarget:
    def __init__(self, name: str, smoke_result: object | None) -> None:
        self.name = name
        self._smoke_result = smoke_result
        self.calls: list[object] = []

    def smoke_test(self, prepared: object) -> object | None:
        self.calls.append(prepared)
        return self._smoke_result


class TestExportRuntimeTargetPaths:
    def test_vllm_target_success_prints_launch_config_and_smoke(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()
        captured: dict[str, Any] = {}
        smoke = SimpleNamespace(ok=True, detail="vllm smoke ok")
        fake_target = _FakeTarget("vllm", smoke)

        _patch_text_export_runtime(monkeypatch)
        monkeypatch.setattr("dlm.export.targets.resolve_target", lambda name: fake_target)

        def _prepare(**kwargs: object) -> object:
            captured.update(kwargs)
            export_dir = tmp_path / "exports" / "vllm"
            launch = export_dir / "vllm_launch.sh"
            config = export_dir / "vllm_config.json"
            return SimpleNamespace(
                export_dir=export_dir,
                launch_script_path=launch,
                config_path=config,
            )

        monkeypatch.setattr("dlm.export.targets.prepare_vllm_export", _prepare)
        monkeypatch.setattr(
            "dlm.export.targets.finalize_vllm_export",
            lambda **kwargs: tmp_path / "exports" / "vllm" / "export_manifest.json",
        )

        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "export",
                str(doc),
                "--target",
                "vllm",
                "--name",
                "served-demo",
                "--quant",
                "Q4_K_M",
                "--merged",
                "--dequantize",
                "--no-template",
                "--skip-ollama",
                "--no-imatrix",
                "--draft",
                "qwen2.5:0.5b",
            ],
        )

        assert result.exit_code == 0, result.output
        text = _joined_output(result)
        assert "ignoring flags not applicable to `--target vllm`" in text
        assert "--quant" in text
        assert "--merged" in text
        assert "--dequantize" in text
        assert "--no-template" in text
        assert "--skip-ollama" in text
        assert "--no-imatrix" in text
        assert "--draft" in text
        assert "target: vllm" in text
        assert "launch: vllm_launch.sh" in text
        assert "config: vllm_config.json" in text
        assert "manifest: export_manifest.json" in text
        assert "smoke: vllm smoke ok" in text
        assert captured["served_model_name"] == "served-demo"
        assert captured["training_sequence_len"] == 2048
        assert fake_target.calls

    def test_vllm_target_prepare_error_exits_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()

        _patch_text_export_runtime(monkeypatch)
        monkeypatch.setattr(
            "dlm.export.targets.resolve_target",
            lambda name: _FakeTarget("vllm", None),
        )
        monkeypatch.setattr(
            "dlm.export.targets.prepare_vllm_export",
            lambda **kwargs: (_ for _ in ()).throw(ExportError("vllm prepare failed")),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc), "--target", "vllm"],
        )

        assert result.exit_code == 1, result.output
        assert "vllm prepare failed" in _joined_output(result)

    def test_vllm_target_smoke_failure_exits_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()
        fake_target = _FakeTarget("vllm", SimpleNamespace(ok=False, detail="vllm smoke failed"))

        _patch_text_export_runtime(monkeypatch)
        monkeypatch.setattr("dlm.export.targets.resolve_target", lambda name: fake_target)
        monkeypatch.setattr(
            "dlm.export.targets.prepare_vllm_export",
            lambda **kwargs: SimpleNamespace(
                export_dir=tmp_path / "exports" / "vllm",
                launch_script_path=tmp_path / "exports" / "vllm" / "vllm_launch.sh",
                config_path=tmp_path / "exports" / "vllm" / "vllm_config.json",
            ),
        )
        monkeypatch.setattr(
            "dlm.export.targets.finalize_vllm_export",
            lambda **kwargs: tmp_path / "exports" / "vllm" / "export_manifest.json",
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc), "--target", "vllm"],
        )

        assert result.exit_code == 1, result.output
        text = _joined_output(result)
        assert "vllm smoke failed" in text
        assert "re-run with `--no-smoke`" in text

    def test_mlx_target_success_prints_launch_manifest_and_smoke(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()
        captured: dict[str, Any] = {}
        smoke = SimpleNamespace(ok=True, detail="mlx smoke ok")
        fake_target = _FakeTarget("mlx-serve", smoke)

        _patch_text_export_runtime(monkeypatch)
        monkeypatch.setattr("dlm.export.targets.resolve_target", lambda name: fake_target)

        def _prepare(**kwargs: object) -> object:
            captured.update(kwargs)
            export_dir = tmp_path / "exports" / "mlx-serve"
            launch = export_dir / "mlx_serve_launch.sh"
            return SimpleNamespace(
                export_dir=export_dir,
                launch_script_path=launch,
            )

        monkeypatch.setattr("dlm.export.targets.prepare_mlx_serve_export", _prepare)
        monkeypatch.setattr(
            "dlm.export.targets.finalize_mlx_serve_export",
            lambda **kwargs: tmp_path / "exports" / "mlx-serve" / "export_manifest.json",
        )

        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "export",
                str(doc),
                "--target",
                "mlx-serve",
                "--name",
                "ignored-name",
                "--quant",
                "Q4_K_M",
                "--merged",
                "--dequantize",
                "--no-template",
                "--skip-ollama",
                "--no-imatrix",
                "--draft",
                "qwen2.5:0.5b",
            ],
        )

        assert result.exit_code == 0, result.output
        text = _joined_output(result)
        assert "ignoring flags not applicable to `--target mlx-serve`" in text
        assert "--name" in text
        assert "--quant" in text
        assert "--merged" in text
        assert "--dequantize" in text
        assert "--no-template" in text
        assert "--skip-ollama" in text
        assert "--no-imatrix" in text
        assert "--draft" in text
        assert "target: mlx-serve" in text
        assert "launch: mlx_serve_launch.sh" in text
        assert "manifest: export_manifest.json" in text
        assert "smoke: mlx smoke ok" in text
        assert captured["adapter_name"] is None
        assert captured["adapter_path_override"] is None
        assert fake_target.calls

    def test_mlx_target_prepare_error_exits_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()

        _patch_text_export_runtime(monkeypatch)
        monkeypatch.setattr(
            "dlm.export.targets.resolve_target",
            lambda name: _FakeTarget("mlx-serve", None),
        )
        monkeypatch.setattr(
            "dlm.export.targets.prepare_mlx_serve_export",
            lambda **kwargs: (_ for _ in ()).throw(ExportError("mlx prepare failed")),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc), "--target", "mlx-serve"],
        )

        assert result.exit_code == 1, result.output
        assert "mlx prepare failed" in _joined_output(result)

    def test_mlx_target_smoke_failure_exits_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()
        fake_target = _FakeTarget("mlx-serve", SimpleNamespace(ok=False, detail="mlx smoke failed"))

        _patch_text_export_runtime(monkeypatch)
        monkeypatch.setattr("dlm.export.targets.resolve_target", lambda name: fake_target)
        monkeypatch.setattr(
            "dlm.export.targets.prepare_mlx_serve_export",
            lambda **kwargs: SimpleNamespace(
                export_dir=tmp_path / "exports" / "mlx-serve",
                launch_script_path=tmp_path / "exports" / "mlx-serve" / "mlx_serve_launch.sh",
            ),
        )
        monkeypatch.setattr(
            "dlm.export.targets.finalize_mlx_serve_export",
            lambda **kwargs: tmp_path / "exports" / "mlx-serve" / "export_manifest.json",
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc), "--target", "mlx-serve"],
        )

        assert result.exit_code == 1, result.output
        text = _joined_output(result)
        assert "mlx smoke failed" in text
        assert "re-run with `--no-smoke`" in text
