"""Unit coverage for modality-aware export dispatch."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dlm.base_models import BASE_MODELS
from dlm.export.arch_probe import ArchProbeResult, SupportLevel
from dlm.export.dispatch import (
    DispatchResult,
    _load_processor_or_raise,
    dispatch_audio_export,
    dispatch_vl_export,
    emit_vl_snapshot,
)
from dlm.export.errors import (
    ExportError,
    ProcessorLoadError,
    VendoringError,
    VlGgufUnsupportedError,
)

_VL_SPEC = BASE_MODELS["qwen2-vl-2b-instruct"]
_AUDIO_SPEC = BASE_MODELS["qwen2-audio-7b-instruct"]


def _snapshot_result(tmp_path: Path, dirname: str) -> object:
    export_dir = tmp_path / dirname
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = export_dir / "export_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    adapter_dir = export_dir / "adapter"
    adapter_dir.mkdir()
    artifact = export_dir / "artifact.txt"
    artifact.write_text("ok", encoding="utf-8")
    return SimpleNamespace(
        export_dir=export_dir,
        manifest_path=manifest_path,
        adapter_dir=adapter_dir,
        artifacts=[artifact],
    )


class TestLoadProcessorOrRaise:
    def test_wraps_loader_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "dlm.train.loader.load_processor",
            lambda spec: (_ for _ in ()).throw(RuntimeError("missing cache")),
        )

        with pytest.raises(ProcessorLoadError, match="missing cache"):
            _load_processor_or_raise(_VL_SPEC)


class TestEmitVlSnapshot:
    def test_emits_snapshot_and_warns_about_gguf_only_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        processor = object()
        monkeypatch.setattr("dlm.train.loader.load_processor", lambda spec: processor)
        monkeypatch.setattr(
            "dlm.export.vl_snapshot.run_vl_snapshot_export",
            lambda store, spec, *, adapter_name, processor: _snapshot_result(
                tmp_path, "vl-snapshot"
            ),
        )

        result = emit_vl_snapshot(
            store=object(),
            spec=_VL_SPEC,
            adapter_name="named",
            quant="Q4_K_M",
            merged=True,
            adapter_mix_raw="tone:0.5",
        )

        assert result.extras["path"] == "hf-snapshot"
        assert any("ignoring GGUF-only flags" in line for line in result.banner_lines)
        assert any("HF snapshot written" in line for line in result.banner_lines)

    def test_skip_warning_suppresses_flag_banner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("dlm.train.loader.load_processor", lambda spec: object())
        monkeypatch.setattr(
            "dlm.export.vl_snapshot.run_vl_snapshot_export",
            lambda store, spec, *, adapter_name, processor: _snapshot_result(
                tmp_path, "vl-snapshot-skip-warning"
            ),
        )

        result = emit_vl_snapshot(
            store=object(),
            spec=_VL_SPEC,
            adapter_name=None,
            quant="Q4_K_M",
            merged=True,
            adapter_mix_raw="tone:0.5",
            skip_gguf_flag_warning=True,
        )

        assert not any("ignoring GGUF-only flags" in line for line in result.banner_lines)


class TestDispatchVlExport:
    def test_probe_vendoring_failure_falls_back_to_snapshot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = DispatchResult(
            export_dir=Path("/tmp/vl"),
            manifest_path=Path("/tmp/vl/export_manifest.json"),
            artifacts=[],
            banner_lines=["snapshot"],
            extras={"path": "hf-snapshot"},
        )
        monkeypatch.setattr(
            "dlm.export.arch_probe.probe_gguf_arch",
            lambda architecture: (_ for _ in ()).throw(VendoringError("missing submodule")),
        )
        monkeypatch.setattr("dlm.export.dispatch.emit_vl_snapshot", lambda **kwargs: expected)

        result = dispatch_vl_export(
            store=object(),
            spec=_VL_SPEC,
            adapter_name=None,
            quant=None,
            merged=False,
            adapter_mix_raw=None,
        )

        assert result.banner_lines[0].startswith(
            "[yellow]export:[/yellow] llama.cpp probe unavailable"
        )
        assert result.banner_lines[-1] == "snapshot"

    @pytest.mark.parametrize(
        ("support", "expected_text"),
        [
            (SupportLevel.UNSUPPORTED, "is not covered by the vendored llama.cpp"),
            (SupportLevel.PARTIAL, "has PARTIAL llama.cpp coverage"),
        ],
    )
    def test_unsupported_or_partial_verdicts_fall_back_to_snapshot(
        self,
        monkeypatch: pytest.MonkeyPatch,
        support: SupportLevel,
        expected_text: str,
    ) -> None:
        verdict = ArchProbeResult(
            arch_class=_VL_SPEC.architecture,
            support=support,
            reason="probe result",
            llama_cpp_tag="b1234",
        )
        expected = DispatchResult(
            export_dir=Path("/tmp/vl"),
            manifest_path=Path("/tmp/vl/export_manifest.json"),
            artifacts=[],
            banner_lines=["snapshot"],
            extras={"path": "hf-snapshot"},
        )
        monkeypatch.setattr("dlm.export.arch_probe.probe_gguf_arch", lambda architecture: verdict)
        monkeypatch.setattr("dlm.export.dispatch.emit_vl_snapshot", lambda **kwargs: expected)

        result = dispatch_vl_export(
            store=object(),
            spec=_VL_SPEC,
            adapter_name=None,
            quant=None,
            merged=False,
            adapter_mix_raw=None,
        )

        assert expected_text in result.banner_lines[0]
        assert result.banner_lines[-1] == "snapshot"

    def test_supported_without_context_falls_back_to_snapshot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        verdict = ArchProbeResult(
            arch_class=_VL_SPEC.architecture,
            support=SupportLevel.SUPPORTED,
            reason="probe result",
            llama_cpp_tag="b1234",
        )
        expected = DispatchResult(
            export_dir=Path("/tmp/vl"),
            manifest_path=Path("/tmp/vl/export_manifest.json"),
            artifacts=[],
            banner_lines=["snapshot"],
            extras={"path": "hf-snapshot"},
        )
        monkeypatch.setattr("dlm.export.arch_probe.probe_gguf_arch", lambda architecture: verdict)
        monkeypatch.setattr("dlm.export.dispatch.emit_vl_snapshot", lambda **kwargs: expected)

        result = dispatch_vl_export(
            store=object(),
            spec=_VL_SPEC,
            adapter_name="named",
            quant="Q4_K_M",
            merged=False,
            adapter_mix_raw=None,
            gguf_emission_context=None,
        )

        assert "without GGUF plan context" in result.banner_lines[0]

    def test_supported_verdict_returns_vl_gguf_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        verdict = ArchProbeResult(
            arch_class=_VL_SPEC.architecture,
            support=SupportLevel.SUPPORTED,
            reason="probe result",
            llama_cpp_tag="b5678",
        )
        export_dir = tmp_path / "vl-gguf"
        export_dir.mkdir()
        manifest_path = export_dir / "export_manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        gguf_path = export_dir / "model.gguf"
        gguf_path.write_bytes(b"gguf")
        modelfile_path = export_dir / "Modelfile"
        modelfile_path.write_text("FROM base", encoding="utf-8")
        mmproj_path = export_dir / "mmproj.gguf"
        mmproj_path.write_bytes(b"mmproj")

        monkeypatch.setattr("dlm.export.arch_probe.probe_gguf_arch", lambda architecture: verdict)
        monkeypatch.setattr(
            "dlm.export.vl_gguf.run_vl_gguf_export",
            lambda *args, **kwargs: SimpleNamespace(
                export_dir=export_dir,
                manifest_path=manifest_path,
                gguf_path=gguf_path,
                modelfile_path=modelfile_path,
                mmproj_path=mmproj_path,
                quant="Q4_K_M",
                llama_cpp_tag="b5678",
                artifacts=[gguf_path, modelfile_path],
            ),
        )

        result = dispatch_vl_export(
            store=object(),
            spec=_VL_SPEC,
            adapter_name=None,
            quant="Q4_K_M",
            merged=True,
            adapter_mix_raw=None,
            gguf_emission_context={
                "plan": object(),
                "cached_base_dir": tmp_path / "cache",
                "source_dlm_path": tmp_path / "doc.dlm",
                "dlm_version": "test",
                "training_sequence_len": 1024,
            },
        )

        assert result.extras["path"] == "vl-gguf"
        assert result.extras["gguf_path"] == gguf_path
        assert any(
            "attempting single-file VL GGUF emission" in line for line in result.banner_lines
        )
        assert any("VL GGUF written" in line for line in result.banner_lines)

    @pytest.mark.parametrize(
        "error",
        [
            VlGgufUnsupportedError("plan refused"),
            VendoringError("missing binary"),
            ExportError("subprocess failed"),
        ],
    )
    def test_supported_verdict_falls_back_after_gguf_failure(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        verdict = ArchProbeResult(
            arch_class=_VL_SPEC.architecture,
            support=SupportLevel.SUPPORTED,
            reason="probe result",
            llama_cpp_tag="b1234",
        )
        expected = DispatchResult(
            export_dir=Path("/tmp/vl"),
            manifest_path=Path("/tmp/vl/export_manifest.json"),
            artifacts=[],
            banner_lines=["snapshot"],
            extras={"path": "hf-snapshot"},
        )
        monkeypatch.setattr("dlm.export.arch_probe.probe_gguf_arch", lambda architecture: verdict)
        monkeypatch.setattr(
            "dlm.export.vl_gguf.run_vl_gguf_export",
            lambda *args, **kwargs: (_ for _ in ()).throw(error),
        )
        monkeypatch.setattr("dlm.export.dispatch.emit_vl_snapshot", lambda **kwargs: expected)

        result = dispatch_vl_export(
            store=object(),
            spec=_VL_SPEC,
            adapter_name=None,
            quant="Q4_K_M",
            merged=True,
            adapter_mix_raw=None,
            gguf_emission_context={
                "plan": object(),
                "cached_base_dir": Path("/tmp/cache"),
            },
        )

        assert "falling back to HF-snapshot" in "\n".join(result.banner_lines)


class TestDispatchAudioExport:
    def test_audio_export_uses_snapshot_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("dlm.train.loader.load_processor", lambda spec: object())
        monkeypatch.setattr(
            "dlm.export.audio_snapshot.run_audio_snapshot_export",
            lambda store, spec, *, adapter_name, processor: _snapshot_result(
                tmp_path, "audio-snapshot"
            ),
        )

        result = dispatch_audio_export(
            store=object(),
            spec=_AUDIO_SPEC,
            adapter_name="named",
            quant="Q4_K_M",
            merged=True,
            adapter_mix_raw="tone:0.5",
        )

        assert result.extras["path"] == "audio-snapshot"
        assert any("audio-language" in line for line in result.banner_lines)
        assert any("ignoring GGUF-only flags" in line for line in result.banner_lines)
        assert any("HF audio snapshot written" in line for line in result.banner_lines)
