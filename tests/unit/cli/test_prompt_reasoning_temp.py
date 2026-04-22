"""`dlm prompt` default temperature honors reasoning-tuned base specs."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm.base_models import BaseModelSpec
from dlm.cli.app import app


def _write_doc(path: Path, *, base_model: str = "reasoner-1b") -> None:
    path.write_text(
        f"---\ndlm_id: 01HZ4X7TGZM3J1A2B3C4D5E6F7\nbase_model: {base_model}\n---\nbody\n",
        encoding="utf-8",
    )


def _spec(*, reasoning_tuned: bool) -> BaseModelSpec:
    return BaseModelSpec.model_validate(
        {
            "key": "reasoner-1b",
            "hf_id": "org/reasoner-1b",
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
            "reasoning_tuned": reasoning_tuned,
        }
    )


class _FakeBackend:
    def __init__(self, sink: dict[str, object]) -> None:
        self._sink = sink

    def load(self, spec: object, store: object, adapter_name: str | None = None) -> None:
        return None

    def generate(self, query: str, **kwargs: object) -> str:
        self._sink["query"] = query
        self._sink["kwargs"] = kwargs
        return "ok"


class TestPromptReasoningTemperature:
    def test_reasoning_tuned_base_uses_cooler_default_when_temp_omitted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = tmp_path / "doc.dlm"
        _write_doc(doc)
        captured: dict[str, object] = {}
        runner = CliRunner()

        monkeypatch.setattr(
            "dlm.base_models.resolve", lambda *args, **kwargs: _spec(reasoning_tuned=True)
        )
        monkeypatch.setattr(
            "dlm.hardware.doctor", lambda: type("R", (), {"capabilities": object()})()
        )
        monkeypatch.setattr(
            "dlm.inference.backends.select_backend", lambda *args, **kwargs: "pytorch"
        )
        monkeypatch.setattr(
            "dlm.inference.backends.build_backend",
            lambda *args, **kwargs: _FakeBackend(captured),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "prompt", str(doc), "hello"],
        )
        assert result.exit_code == 0, result.output
        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs["temperature"] == pytest.approx(0.6)

    def test_explicit_temp_overrides_reasoning_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = tmp_path / "doc.dlm"
        _write_doc(doc)
        captured: dict[str, object] = {}
        runner = CliRunner()

        monkeypatch.setattr(
            "dlm.base_models.resolve", lambda *args, **kwargs: _spec(reasoning_tuned=True)
        )
        monkeypatch.setattr(
            "dlm.hardware.doctor", lambda: type("R", (), {"capabilities": object()})()
        )
        monkeypatch.setattr(
            "dlm.inference.backends.select_backend", lambda *args, **kwargs: "pytorch"
        )
        monkeypatch.setattr(
            "dlm.inference.backends.build_backend",
            lambda *args, **kwargs: _FakeBackend(captured),
        )

        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "prompt",
                str(doc),
                "hello",
                "--temp",
                "0.9",
            ],
        )
        assert result.exit_code == 0, result.output
        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs["temperature"] == pytest.approx(0.9)
