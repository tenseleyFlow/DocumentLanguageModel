"""`run_export` orchestration — mocked subprocess + real filesystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dlm.base_models import BASE_MODELS
from dlm.export import ExportPlan, ExportResult, run_export
from dlm.export.errors import ExportError, UnsafeMergeError
from dlm.export.manifest import load_export_manifest
from dlm.store.manifest import Manifest, save_manifest
from dlm.store.paths import for_dlm

_SPEC = BASE_MODELS["smollm2-135m"]


class _SubprocessRecorder:
    """Captures commands + fakes subprocess output (writes expected files)."""

    def __init__(self, export_dir: Path) -> None:
        self.commands: list[list[str]] = []
        self.export_dir = export_dir

    def __call__(self, cmd: Any) -> Any:
        cmd_list = list(cmd)
        self.commands.append(cmd_list)

        # `convert_hf_to_gguf.py ... --outfile <out> ...` → write stub output.
        if any(a.endswith("convert_hf_to_gguf.py") for a in cmd_list):
            out = _flag_value(cmd_list, "--outfile")
            if out is not None:
                Path(out).write_bytes(b"fake gguf bytes (hf)")

        # `llama-quantize <in> <out> <QUANT>` — out is argv[-2].
        if cmd_list and ("llama-quantize" in cmd_list[0] or cmd_list[0].endswith("quantize")):
            out = Path(cmd_list[-2])
            out.write_bytes(b"fake quantized bytes")

        # `convert_lora_to_gguf.py <adapter> --outfile <out> ...`
        if any(a.endswith("convert_lora_to_gguf.py") for a in cmd_list):
            out = _flag_value(cmd_list, "--outfile")
            if out is not None:
                Path(out).write_bytes(b"fake lora gguf")

        return None


def _flag_value(argv: list[str], flag: str) -> str | None:
    try:
        idx = argv.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(argv):
        return None
    return argv[idx + 1]


def _setup_store(tmp_path: Path, *, use_qlora: bool = False) -> tuple[Path, Any, Path]:
    """Build a store + fake adapter + fake vendor tree.

    Returns `(cached_base_dir, store, vendor_override)` so tests can
    thread the vendor override through `run_export`.
    """
    store = for_dlm("01TEST", home=tmp_path)
    store.ensure_layout()
    save_manifest(store.manifest, Manifest(dlm_id="01TEST", base_model=_SPEC.key))

    adapter = store.adapter_version(1)
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": _SPEC.hf_id, "peft_type": "LORA"})
    )
    (adapter / "tokenizer_config.json").write_text(
        json.dumps({"vocab_size": 32000, "chat_template": "{{messages}}"})
    )
    (adapter / "training_run.json").write_text(json.dumps({"use_qlora": use_qlora}))
    store.set_current_adapter(adapter)

    # Fake HF cache dir — just needs to exist; the mock subprocess
    # doesn't actually read it.
    cached_base = tmp_path / "cache" / "base"
    cached_base.mkdir(parents=True)

    # Fake llama.cpp vendor tree — populated with empty stub files so
    # `vendoring.convert_*_py` + `llama_quantize_bin` resolve.
    vendor = tmp_path / "vendor" / "llama.cpp"
    vendor.mkdir(parents=True)
    (vendor / "convert_hf_to_gguf.py").write_text("# mock")
    (vendor / "convert_lora_to_gguf.py").write_text("# mock")
    bin_dir = vendor / "build" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "llama-quantize").write_text("# mock")
    (vendor / "VERSION").write_text("b9999\n")

    return cached_base, store, vendor


class TestHappyPath:
    def test_unmerged_export_emits_base_and_adapter(self, tmp_path: Path) -> None:
        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q4_K_M", merged=False)
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))

        result = run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=recorder,
            vendor_override=vendor,
            skip_ollama=True,
        )

        assert isinstance(result, ExportResult)
        assert (result.export_dir / f"base.{plan.quant}.gguf").exists()
        assert (result.export_dir / "adapter.gguf").exists()
        assert result.manifest_path.exists()
        # 3 subprocess calls on the first export: convert_hf + quantize + convert_lora.
        assert len(recorder.commands) == 3

    def test_export_manifest_contents(self, tmp_path: Path) -> None:
        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q5_K_M", ollama_name="mydoc:latest")
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))
        result = run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=recorder,
            vendor_override=vendor,
            skip_ollama=True,
        )

        em = load_export_manifest(result.export_dir)
        assert em.quant == "Q5_K_M"
        assert em.ollama_name == "mydoc:latest"
        assert em.base_model_hf_id == _SPEC.hf_id
        assert em.base_model_revision == _SPEC.revision
        assert em.adapter_version == 1
        assert len(em.artifacts) == 2  # base + adapter
        assert all(a.sha256 and a.size_bytes > 0 for a in em.artifacts)
        assert em.llama_cpp_tag == "b9999"


class TestCaching:
    def test_second_export_skips_base_conversion(self, tmp_path: Path) -> None:
        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q4_K_M")

        recorder1 = _SubprocessRecorder(store.export_quant_dir(plan.quant))
        r1 = run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=recorder1,
            vendor_override=vendor,
            skip_ollama=True,
        )
        assert r1.cached is False
        assert len(recorder1.commands) == 3  # convert_hf + quantize + convert_lora

        recorder2 = _SubprocessRecorder(store.export_quant_dir(plan.quant))
        r2 = run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=recorder2,
            vendor_override=vendor,
            skip_ollama=True,
        )
        assert r2.cached is True
        # Only the adapter conversion runs on the cached path.
        assert len(recorder2.commands) == 1
        assert any("convert_lora_to_gguf.py" in str(a) for a in recorder2.commands[0])


class TestMergeGate:
    def test_qlora_merge_without_dequantize_raises(self, tmp_path: Path) -> None:
        cached_base, store, vendor = _setup_store(tmp_path, use_qlora=True)
        plan = ExportPlan(merged=True, dequantize_confirmed=False)
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))

        with pytest.raises(UnsafeMergeError):
            run_export(
                store,
                _SPEC,
                plan,
                cached_base_dir=cached_base,
                subprocess_runner=recorder,
                vendor_override=vendor,
            )
        # No subprocess should have launched on the safety-gate path.
        assert recorder.commands == []


class TestMissingAdapter:
    def test_no_current_adapter_raises(self, tmp_path: Path) -> None:
        store = for_dlm("01TEST", home=tmp_path)
        store.ensure_layout()
        save_manifest(store.manifest, Manifest(dlm_id="01TEST", base_model=_SPEC.key))
        cached_base = tmp_path / "cache"
        cached_base.mkdir()

        with pytest.raises(ExportError, match="no current adapter"):
            run_export(
                store,
                _SPEC,
                ExportPlan(),
                cached_base_dir=cached_base,
                subprocess_runner=lambda _cmd: None,
            )


class TestManifestAppend:
    def test_exports_list_grows(self, tmp_path: Path) -> None:
        from dlm.store.manifest import load_manifest

        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q4_K_M", ollama_name="tag")
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))
        run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=recorder,
            vendor_override=vendor,
            skip_ollama=True,
        )

        manifest = load_manifest(store.manifest)
        assert len(manifest.exports) == 1
        export = manifest.exports[0]
        assert export.quant == "Q4_K_M"
        assert export.ollama_name == "tag"
        assert export.merged is False
        assert export.base_gguf_sha256
        assert export.adapter_gguf_sha256

    def test_append_holds_store_lock(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Concurrent export summary-append must serialize on the store lock (audit-04 B3)."""
        from dlm.export import runner as runner_mod
        from dlm.store.errors import LockHeldError
        from dlm.store.lock import exclusive
        from dlm.store.manifest import load_manifest

        # Shrink the append lock timeout so the held-lock branch surfaces quickly.
        monkeypatch.setattr(runner_mod, "_APPEND_LOCK_TIMEOUT", 0.2)

        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q4_K_M", ollama_name="tag")
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))

        # Hold the store lock from a "peer process"; the run_export
        # append should then time out rather than racing the read.
        with exclusive(store.lock, timeout=1.0), pytest.raises(LockHeldError):
            run_export(
                store,
                _SPEC,
                plan,
                cached_base_dir=cached_base,
                subprocess_runner=recorder,
                vendor_override=vendor,
                skip_ollama=True,
            )

        # Peer released → no export summary landed (we errored before save).
        manifest = load_manifest(store.manifest)
        assert len(manifest.exports) == 0
