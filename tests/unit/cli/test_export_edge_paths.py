"""Focused early-branch coverage for `dlm export`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from dlm.base_models import BaseModelSpec
from dlm.base_models.errors import GatedModelError
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


def _spec(*, key: str = "demo-1b", modality: str = "text") -> BaseModelSpec:
    payload: dict[str, object] = {
        "key": key,
        "hf_id": f"org/{key}",
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
        "modality": modality,
    }
    if modality == "vision-language":
        payload["vl_preprocessor_plan"] = {
            "target_size": [224, 224],
            "image_token": "<image>",
            "num_image_tokens": 256,
        }
    elif modality == "audio-language":
        payload["audio_preprocessor_plan"] = {
            "sample_rate": 16000,
            "audio_token": "<audio>",
            "num_audio_tokens": 64,
            "max_length_seconds": 30.0,
        }
    return BaseModelSpec.model_validate(payload)


def _patch_export_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    spec: BaseModelSpec | None = None,
    dispatch: object | None = None,
) -> None:
    monkeypatch.setattr(
        "dlm.base_models.resolve",
        lambda *args, **kwargs: spec or _spec(),
    )
    monkeypatch.setattr(
        "dlm.modality.modality_for",
        lambda model_spec: (
            dispatch
            or SimpleNamespace(
                accepts_images=model_spec.modality == "vision-language",
                accepts_audio=model_spec.modality == "audio-language",
            )
        ),
    )


class TestExportEdgePaths:
    def test_gate_fallback_banner_prints_before_gated_base_refusal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()

        monkeypatch.setattr(
            "dlm.export.gate_fallback.resolve_and_announce",
            lambda store, parsed: SimpleNamespace(
                entries=[("knowledge", 0.7), ("tone", 0.3)],
                banner_lines=["[yellow]gate:[/yellow] using learned adapter prior"],
            ),
        )
        monkeypatch.setattr(
            "dlm.base_models.resolve",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                GatedModelError("org/gated-base", "https://example.test/license")
            ),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc)],
        )

        assert result.exit_code == 1, result.output
        text = _joined_output(result)
        assert "using learned adapter prior" in text
        assert "review the license at: https://example.test/license" in text
        assert "accept via `dlm train --i-accept-license` before exporting." in text

    @pytest.mark.parametrize(
        ("target", "modality", "needle"),
        [
            (
                "vllm",
                "audio-language",
                "--target vllm is not wired for audio-language documents yet",
            ),
            (
                "mlx-serve",
                "audio-language",
                "--target mlx-serve is not wired for audio-language documents yet",
            ),
            (
                "vllm",
                "vision-language",
                "--target vllm is not wired for vision-language documents yet",
            ),
            (
                "mlx-serve",
                "vision-language",
                "--target mlx-serve is not wired for vision-language documents yet",
            ),
        ],
    )
    def test_runtime_targets_refuse_unsupported_modalities(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        target: str,
        modality: str,
        needle: str,
    ) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()

        _patch_export_runtime(
            monkeypatch, spec=_spec(key=f"{target}-{modality}", modality=modality)
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc), "--target", target],
        )

        assert result.exit_code == 2, result.output
        assert needle in _joined_output(result)

    def test_audio_dispatch_export_error_maps_to_exit_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()

        class _AudioDispatch:
            accepts_images = False
            accepts_audio = True

            def dispatch_export(self, **kwargs: object) -> object:
                raise ExportError("audio snapshot failed")

        _patch_export_runtime(
            monkeypatch,
            spec=_spec(key="audio-demo", modality="audio-language"),
            dispatch=_AudioDispatch(),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc)],
        )

        assert result.exit_code == 1, result.output
        assert "audio snapshot failed" in _joined_output(result)

    def test_vl_dispatch_export_error_maps_to_exit_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()

        class _VlDispatch:
            accepts_images = True
            accepts_audio = False

            def dispatch_export(self, **kwargs: object) -> object:
                raise ExportError("vl snapshot failed")

        _patch_export_runtime(
            monkeypatch,
            spec=_spec(key="vl-demo", modality="vision-language"),
            dispatch=_VlDispatch(),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc)],
        )

        assert result.exit_code == 1, result.output
        assert "vl snapshot failed" in _joined_output(result)

    def test_invalid_export_plan_value_exits_2(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()

        _patch_export_runtime(monkeypatch)
        monkeypatch.setattr(
            "dlm.export.resolve_export_plan",
            lambda **kwargs: (_ for _ in ()).throw(ValueError("bad export plan")),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc)],
        )

        assert result.exit_code == 2, result.output
        assert "bad export plan" in _joined_output(result)
