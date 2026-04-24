"""Focused `dlm prompt` edge coverage for the remaining text/VL/audio branches."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from dlm.base_models import BaseModelSpec
from dlm.cli.app import app


def _write_doc(path: Path, *, base_model: str = "demo-1b") -> None:
    path.write_text(
        f"---\ndlm_id: 01HZ4X7TGZM3J1A2B3C4D5E6F7\nbase_model: {base_model}\n---\nbody\n",
        encoding="utf-8",
    )


def _joined_output(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(text.split())


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


def _patch_prompt_runtime(
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
        "dlm.hardware.doctor",
        lambda: SimpleNamespace(capabilities=object()),
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


class TestPromptEdgeBranches:
    def test_invalid_backend_value_exits_2(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        _write_doc(doc)
        runner = CliRunner()

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "prompt", str(doc), "hello", "--backend", "bogus"],
        )

        assert result.exit_code == 2, result.output
        assert "--backend must be" in _joined_output(result)

    def test_gated_base_without_recorded_acceptance_exits_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlm.base_models.errors import GatedModelError

        doc = tmp_path / "doc.dlm"
        _write_doc(doc, base_model="gated-base")
        runner = CliRunner()

        monkeypatch.setattr(
            "dlm.base_models.resolve",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                GatedModelError("org/gated-base", "https://license.example")
            ),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "prompt", str(doc), "hello"],
        )

        assert result.exit_code == 1, result.output
        assert "run `dlm train --i-accept-license` first" in _joined_output(result)

    def test_unsupported_backend_error_exits_2(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlm.inference.backends.select import UnsupportedBackendError

        doc = tmp_path / "doc.dlm"
        _write_doc(doc)
        runner = CliRunner()

        _patch_prompt_runtime(monkeypatch)
        monkeypatch.setattr(
            "dlm.inference.backends.select_backend",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                UnsupportedBackendError("mlx backend unavailable")
            ),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "prompt", str(doc), "hello", "--backend", "mlx"],
        )

        assert result.exit_code == 2, result.output
        assert "mlx backend unavailable" in _joined_output(result)

    def test_verbose_text_prompt_logs_backend_and_generates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = tmp_path / "doc.dlm"
        _write_doc(doc)
        runner = CliRunner()
        captured: dict[str, Any] = {}

        class _FakeBackend:
            def load(self, spec: object, store: object, adapter_name: str | None = None) -> None:
                captured["adapter_name"] = adapter_name

            def generate(self, query: str, **kwargs: object) -> str:
                captured["query"] = query
                captured["kwargs"] = kwargs
                return "ok"

        _patch_prompt_runtime(monkeypatch)
        monkeypatch.setattr(
            "dlm.inference.backends.select_backend",
            lambda *args, **kwargs: "pytorch",
        )
        monkeypatch.setattr(
            "dlm.inference.backends.build_backend",
            lambda *args, **kwargs: _FakeBackend(),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "prompt", str(doc), "hello", "--verbose"],
        )

        assert result.exit_code == 0, result.output
        assert captured["query"] == "hello"
        assert "backend: pytorch" in _joined_output(result)
        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs["top_p"] is None

    def test_missing_adapter_maps_to_exit_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlm.inference import AdapterNotFoundError

        doc = tmp_path / "doc.dlm"
        _write_doc(doc)
        runner = CliRunner()

        class _MissingAdapterBackend:
            def load(self, spec: object, store: object, adapter_name: str | None = None) -> None:
                raise AdapterNotFoundError("missing adapter")

        _patch_prompt_runtime(monkeypatch)
        monkeypatch.setattr(
            "dlm.inference.backends.select_backend",
            lambda *args, **kwargs: "pytorch",
        )
        monkeypatch.setattr(
            "dlm.inference.backends.build_backend",
            lambda *args, **kwargs: _MissingAdapterBackend(),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "prompt", str(doc), "hello"],
        )

        assert result.exit_code == 1, result.output
        assert "missing adapter" in _joined_output(result)

    def test_vision_language_dispatch_branch_invokes_helper(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = tmp_path / "doc.dlm"
        _write_doc(doc, base_model="vl-demo")
        image = tmp_path / "frame.png"
        image.write_bytes(b"\x89PNG fake")
        runner = CliRunner()
        captured: dict[str, Any] = {}

        _patch_prompt_runtime(
            monkeypatch,
            spec=_spec(key="vl-demo", modality="vision-language"),
        )
        monkeypatch.setattr(
            "dlm.cli.commands._dispatch_vl_prompt",
            lambda **kwargs: captured.update(kwargs),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "prompt", str(doc), "hello", "--image", str(image)],
        )

        assert result.exit_code == 0, result.output
        assert captured["query"] == "hello"
        assert captured["image_paths"] == [image]
        assert captured["spec"].key == "vl-demo"

    def test_audio_dispatch_branch_invokes_helper(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = tmp_path / "doc.dlm"
        _write_doc(doc, base_model="audio-demo")
        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"fake wav bytes")
        runner = CliRunner()
        captured: dict[str, Any] = {}

        _patch_prompt_runtime(
            monkeypatch,
            spec=_spec(key="audio-demo", modality="audio-language"),
        )
        monkeypatch.setattr(
            "dlm.cli.commands._dispatch_audio_prompt",
            lambda **kwargs: captured.update(kwargs),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "prompt", str(doc), "hello", "--audio", str(audio)],
        )

        assert result.exit_code == 0, result.output
        assert captured["query"] == "hello"
        assert captured["audio_paths"] == [audio]
        assert captured["spec"].key == "audio-demo"
