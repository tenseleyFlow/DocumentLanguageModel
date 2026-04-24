"""Extra edge coverage for the early validation block in `dlm train`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import dlm.base_models as base_models
from dlm.base_models.errors import GatedModelError
from dlm.cli.app import app
from dlm.cli.scaffold import ScaffoldError
from dlm.doc.errors import DlmParseError


def _write_minimal_dlm(path: Path) -> None:
    path.write_text(
        "---\n"
        "dlm_id: 01TRAINEDGE0000000000000000\n"
        "base_model: smollm2-135m\n"
        "training:\n"
        "  seed: 42\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )


def _parsed_doc(base_model: str = "smollm2-135m") -> object:
    return SimpleNamespace(
        frontmatter=SimpleNamespace(
            base_model=base_model,
            dlm_id="01TRAINEDGE0000000000000000",
            training=SimpleNamespace(sequence_len=2048),
        )
    )


def _resolved_spec(**overrides: Any) -> object:
    defaults: dict[str, Any] = {
        "key": "smollm2-135m",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "modality": "text",
        "params": 135_000_000,
        "effective_context_length": 2048,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestTrainValidationEdges:
    def test_invalid_phase_refused(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(tmp_path), "train", str(doc), "--phase", "bogus"],
        )

        assert result.exit_code == 2, result.output
        assert "--phase must be one of sft|preference|all" in result.output

    def test_resume_and_fresh_refused_together(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(tmp_path), "train", str(doc), "--resume", "--fresh"],
        )

        assert result.exit_code == 2, result.output
        assert "--resume and --fresh are mutually exclusive" in result.output

    def test_invalid_policy_refused(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(tmp_path), "train", str(doc), "--policy", "bogus"],
        )

        assert result.exit_code == 2, result.output
        assert "--policy must be 'permissive' or 'strict'" in result.output

    def test_multi_gpu_exit_code_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        seen: dict[str, object] = {}

        monkeypatch.setattr(
            "dlm.hardware.capabilities.probe",
            lambda: SimpleNamespace(supports_bf16=False),
        )

        def _fake_dispatch(
            gpus: str,
            argv: list[str],
            console: object,
            *,
            mixed_precision: str = "bf16",
        ) -> int | None:
            seen["gpus"] = gpus
            seen["argv"] = argv
            seen["mixed_precision"] = mixed_precision
            return 17

        monkeypatch.setattr("dlm.cli.commands._maybe_dispatch_multi_gpu", _fake_dispatch)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(tmp_path), "train", str(doc), "--gpus", "0,1"],
        )

        assert result.exit_code == 17, result.output
        assert seen["gpus"] == "0,1"
        assert seen["mixed_precision"] == "fp16"

    def test_scaffold_error_exits_cleanly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "corpus"
        target.mkdir()

        def _fake_scaffold(*args: object, **kwargs: object) -> object:
            raise ScaffoldError("bad scaffold", path=target)

        monkeypatch.setattr("dlm.cli.scaffold.scaffold_train_target", _fake_scaffold)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(tmp_path), "train", str(target), "--base", "smollm2-135m"],
        )

        assert result.exit_code == 1, result.output
        assert "bad scaffold" in result.output

    def test_parse_error_exits_cleanly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        monkeypatch.setattr(
            "dlm.doc.parser.parse_file",
            lambda path: (_ for _ in ()).throw(
                DlmParseError("broken frontmatter", path=doc, line=2, col=1)
            ),
        )

        runner = CliRunner()
        result = runner.invoke(app, ["--home", str(tmp_path), "train", str(doc)])

        assert result.exit_code == 1, result.output
        assert "broken frontmatter" in result.output

    def test_gated_base_refusal_surfaces_license_pointer(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        monkeypatch.setattr("dlm.doc.parser.parse_file", lambda path: _parsed_doc("llama-3.2-1b"))

        def _fake_resolve(
            base: str,
            *,
            accept_license: bool = False,
            skip_export_probes: bool = False,
        ) -> object:
            raise GatedModelError(base, "https://example.test/license")

        monkeypatch.setattr(base_models, "resolve", _fake_resolve)

        runner = CliRunner()
        result = runner.invoke(app, ["--home", str(tmp_path), "train", str(doc)])

        assert result.exit_code == 1, result.output
        text = " ".join(result.output.split())
        assert "base model 'llama-3.2-1b' is gated" in text
        assert "https://example.test/license" in text
        assert "--i-accept-license" in text

    def test_doctor_no_plan_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        monkeypatch.setattr("dlm.doc.parser.parse_file", lambda path: _parsed_doc())
        monkeypatch.setattr(base_models, "resolve", lambda *args, **kwargs: _resolved_spec())
        monkeypatch.setattr("dlm.train.distributed.detect_world_size", lambda: 1)
        monkeypatch.setattr(
            "dlm.hardware.doctor",
            lambda **kwargs: SimpleNamespace(plan=None, capabilities=object()),
        )

        runner = CliRunner()
        result = runner.invoke(app, ["--home", str(tmp_path), "train", str(doc)])

        assert result.exit_code == 1, result.output
        assert "no viable training plan for this host" in result.output
