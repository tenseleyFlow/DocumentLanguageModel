"""CLI-level integration: `dlm export` modality-dispatch refusals.

The VL / audio paths route through a snapshot dispatcher that:

1. Probes the vendored llama.cpp for arch support and emits a banner.
2. Warns when GGUF-only flags (--quant / --merged / --adapter-mix)
   were passed — they don't apply to the HF-snapshot path.
3. Loads the processor (HF-heavy; stubbed here) and hands off to
   `run_vl_snapshot_export` / `run_audio_snapshot_export`.

Unit tests cover each piece in isolation. These tests exercise the
full CLI boundary with the HF load stubbed so we can assert that the
dispatcher wiring is correct end-to-end: banner shape, flag warnings,
and the actual snapshot writer is invoked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from dlm.cli.app import app


def _scaffold_vl_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "vl.dlm"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_path / "home"),
            "init",
            str(doc),
            "--base",
            "paligemma-3b-mix-224",
            "--multimodal",
            "--i-accept-license",
        ],
    )
    assert result.exit_code == 0, result.output
    return doc


def _scaffold_audio_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "audio.dlm"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_path / "home"),
            "init",
            str(doc),
            "--base",
            "qwen2-audio-7b-instruct",
            "--audio",
            "--i-accept-license",
        ],
    )
    assert result.exit_code == 0, result.output
    return doc


def _stub_snapshot_export(monkeypatch: Any, snapshot_module: str) -> list[Any]:
    """Replace the heavy HF load + snapshot writer with fakes.

    Returns the per-call recorder so the test can assert the dispatcher
    reached `run_*_snapshot_export` — which is the signal that all
    earlier wiring (probe, banner, flag-warn) ran without erroring.

    `snapshot_module` is `dlm.export.vl_snapshot` or `dlm.export.audio_snapshot`.
    The dispatcher imports `run_*_snapshot_export` locally inside the
    function body, so we patch the symbol on the source module — every
    fresh lookup resolves to our fake.
    """
    calls: list[Any] = []

    def _fake_load_processor(spec: object, **_: object) -> object:
        return object()

    def _fake_runner(store: object, spec: object, **kwargs: object) -> object:
        calls.append((store, spec, kwargs))
        mod = __import__(snapshot_module, fromlist=["*"])
        result_cls_name = (
            "VlSnapshotResult" if snapshot_module.endswith("vl_snapshot") else "AudioSnapshotResult"
        )
        result_cls = getattr(mod, result_cls_name)
        dummy_dir = Path("/tmp")
        return result_cls(
            export_dir=dummy_dir,
            manifest_path=dummy_dir / "snapshot_manifest.json",
            readme_path=dummy_dir / "README.md",
            adapter_dir=dummy_dir / "adapter",
            processor_dir=dummy_dir / "processor",
            artifacts=[],
        )

    monkeypatch.setattr("dlm.train.loader.load_processor", _fake_load_processor)
    runner_name = (
        "run_vl_snapshot_export"
        if snapshot_module.endswith("vl_snapshot")
        else "run_audio_snapshot_export"
    )
    monkeypatch.setattr(f"{snapshot_module}.{runner_name}", _fake_runner)
    return calls


class TestVlExportDispatcher:
    def test_vl_doc_routes_through_hf_snapshot_with_banner(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        doc = _scaffold_vl_doc(tmp_path)
        calls = _stub_snapshot_export(monkeypatch, "dlm.export.vl_snapshot")
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc)],
        )
        assert result.exit_code == 0, result.output
        # Banner names the base + flags the arch-probe verdict. We don't
        # pin the exact support level (it depends on the vendored
        # llama.cpp tag) but one of the three distinct banners must fire.
        # Rich wraps console output at terminal width, so normalize
        # whitespace before the substring check.
        joined = " ".join(result.output.split())
        banner_markers = [
            "is not covered by the vendored llama.cpp",
            "PARTIAL",
            "is SUPPORTED by llama.cpp",
        ]
        assert any(m in joined for m in banner_markers), (
            f"expected a VL banner, got:\n{result.output}"
        )
        assert len(calls) == 1, "run_vl_snapshot_export should have been invoked once"

    def test_vl_export_warns_on_gguf_only_flags(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        doc = _scaffold_vl_doc(tmp_path)
        _stub_snapshot_export(monkeypatch, "dlm.export.vl_snapshot")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "export",
                str(doc),
                "--quant",
                "Q4_K_M",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "ignoring GGUF-only flags" in result.output


class TestAudioExportDispatcher:
    def test_audio_doc_routes_through_hf_snapshot_with_banner(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        doc = _scaffold_audio_doc(tmp_path)
        calls = _stub_snapshot_export(monkeypatch, "dlm.export.audio_snapshot")
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc)],
        )
        assert result.exit_code == 0, result.output
        # Audio dispatcher banner references the audio-snapshot path and
        # that llama.cpp has no audio coverage on its roadmap.
        assert "audio" in result.output.lower()
        assert len(calls) == 1, "run_audio_snapshot_export should have been invoked once"
