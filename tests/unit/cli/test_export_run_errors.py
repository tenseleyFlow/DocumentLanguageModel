"""CLI coverage for generic `run_export(...)` branches."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from dlm.base_models import BaseModelSpec
from dlm.cli.app import app
from dlm.export.errors import ExportError, PreflightError, SubprocessError, UnsafeMergeError
from dlm.export.ollama.errors import (
    OllamaCreateError,
    OllamaError,
    OllamaSmokeError,
    OllamaVersionError,
)


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


def _patch_export_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(
        "dlm.export.targets.resolve_target",
        lambda name: SimpleNamespace(name="ollama"),
    )


class TestExportRunErrors:
    def test_verbose_success_prints_shell_command_and_cached_tag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()
        captured: dict[str, Any] = {}

        _patch_export_runtime(monkeypatch)

        def _run_export(
            store: object,
            spec: object,
            plan: object,
            **kwargs: object,
        ) -> object:
            captured.update(kwargs)
            subprocess_runner = kwargs["subprocess_runner"]
            assert callable(subprocess_runner)
            subprocess_runner(["llama-quantize", "--version"])
            return SimpleNamespace(
                cached=True,
                export_dir=tmp_path / "exports" / "Q4_K_M",
                artifacts=[SimpleNamespace(name="base.gguf"), SimpleNamespace(name="adapter.gguf")],
                target="ollama",
                ollama_name="demo-model",
                ollama_version=1,
                smoke_output_first_line="hello smoke",
            )

        monkeypatch.setattr("dlm.export.run_export", _run_export)
        monkeypatch.setattr(
            "dlm.export.quantize.run_checked", lambda cmd: SimpleNamespace(returncode=0)
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc), "--verbose"],
        )

        assert result.exit_code == 0, result.output
        text = _joined_output(result)
        assert "$ llama-quantize --version" in text
        assert "(cached base)" in text
        assert "ollama: demo-model (v1)" in text
        assert "smoke: hello smoke" in text
        assert captured["cached_base_dir"] == Path("/tmp/base-cache")
        assert captured["target"] == "ollama"

    @pytest.mark.parametrize(
        ("error", "needle"),
        [
            (UnsafeMergeError("needs --dequantize"), "merge:"),
            (
                PreflightError(probe="template", detail="template mismatch"),
                "preflight: template mismatch",
            ),
            (
                SubprocessError(
                    cmd=["llama-quantize"],
                    returncode=3,
                    stderr_tail="quantize failed",
                ),
                "subprocess:",
            ),
            (
                OllamaVersionError(detected=(0, 1, 0), required=(0, 6, 0)),
                "ollama:",
            ),
            (OllamaCreateError(stdout="", stderr="create failed"), "ollama create:"),
            (OllamaSmokeError(stdout="", stderr="smoke failed"), "smoke:"),
            (OllamaError("generic ollama error"), "ollama:"),
            (ExportError("plain export failure"), "export:"),
        ],
    )
    def test_run_export_error_mappings_exit_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        error: Exception,
        needle: str,
    ) -> None:
        doc = _scaffold_doc(tmp_path)
        runner = CliRunner()

        _patch_export_runtime(monkeypatch)
        monkeypatch.setattr(
            "dlm.export.run_export",
            lambda *args, **kwargs: (_ for _ in ()).throw(error),
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc)],
        )

        assert result.exit_code == 1, result.output
        text = _joined_output(result)
        assert needle in text
        if isinstance(error, OllamaSmokeError):
            assert "re-run with `--no-smoke`" in text
