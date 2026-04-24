"""`run_export` orchestration — mocked subprocess + real filesystem."""

from __future__ import annotations

import json
import logging
from datetime import datetime
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
            quantized_out = Path(cmd_list[-2])
            quantized_out.write_bytes(b"fake quantized bytes")

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


def _setup_named_store(tmp_path: Path) -> tuple[Path, Any, Path]:
    cached_base, store, vendor = _setup_store(tmp_path)
    knowledge = store.adapter_version_for("knowledge", 2)
    knowledge.mkdir(parents=True)
    (knowledge / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": _SPEC.hf_id, "peft_type": "LORA"})
    )
    (knowledge / "tokenizer_config.json").write_text(
        json.dumps({"vocab_size": 32000, "chat_template": "{{messages}}"})
    )
    (knowledge / "training_run.json").write_text(json.dumps({"use_qlora": False}))
    store.set_current_adapter_for("knowledge", knowledge)
    return cached_base, store, vendor


def _relative_file_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class TestHappyPath:
    def test_default_ollama_name_lowercases_dlm_id(self) -> None:
        from dlm.export.runner import default_ollama_name

        assert default_ollama_name("01ABCDEF", 7) == "dlm-01abcdef:v0007"

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
            vocab_checker=lambda _a, _g: None,
        )

        assert isinstance(result, ExportResult)
        assert result.target == "ollama"
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
            vocab_checker=lambda _a, _g: None,
        )

        em = load_export_manifest(result.export_dir)
        assert em.target == "ollama"
        assert em.quant == "Q5_K_M"
        assert em.ollama_name == "mydoc:latest"
        assert em.base_model_hf_id == _SPEC.hf_id
        assert em.base_model_revision == _SPEC.revision
        assert em.adapter_version == 1
        assert len(em.artifacts) == 2  # base + adapter
        assert all(a.sha256 and a.size_bytes > 0 for a in em.artifacts)
        assert em.llama_cpp_tag == "b9999"

    def test_non_ollama_target_still_records_manifest_target(self, tmp_path: Path) -> None:
        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q4_K_M")
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))

        result = run_export(
            store,
            _SPEC,
            plan,
            target="llama-server",
            cached_base_dir=cached_base,
            subprocess_runner=recorder,
            vendor_override=vendor,
            skip_ollama=True,
            vocab_checker=lambda _a, _g: None,
        )

        em = load_export_manifest(result.export_dir)
        assert result.target == "llama-server"
        assert em.target == "llama-server"

    def test_explicit_ollama_target_matches_default_export_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fixed_now = datetime(2026, 4, 23, 12, 0, 0)
        fixed_header = "\n".join(
            [
                "# Generated by dlm 0.1.0 on 2026-04-23T12:00:00",
                "# dlm_id: 01TEST",
                "# adapter_version: 1",
                f"# base_model: {_SPEC.key} (revision {_SPEC.revision})",
                "# quant: Q4_K_M",
                "# merged: False",
            ]
        )
        monkeypatch.setattr("dlm.export.runner.utc_now", lambda: fixed_now)
        monkeypatch.setattr("dlm.export.ollama.modelfile.build_header", lambda **_: fixed_header)

        cached_base_default, store_default, vendor_default = _setup_store(tmp_path / "default")
        cached_base_explicit, store_explicit, vendor_explicit = _setup_store(tmp_path / "explicit")
        plan = ExportPlan(quant="Q4_K_M", ollama_name="mydoc:latest")
        create_calls: list[tuple[str, str]] = []
        run_calls: list[str] = []

        def _fake_create(*, name: str, modelfile_path: Path, cwd: Path) -> str:
            create_calls.append((name, str(modelfile_path.relative_to(cwd))))
            return "created"

        def _fake_run(*, name: str) -> str:
            run_calls.append(name)
            return "hello from ollama\nsecond line"

        result_default = run_export(
            store_default,
            _SPEC,
            plan,
            cached_base_dir=cached_base_default,
            subprocess_runner=_SubprocessRecorder(store_default.export_quant_dir(plan.quant)),
            vendor_override=vendor_default,
            ollama_create_runner=_fake_create,
            ollama_run_runner=_fake_run,
            vocab_checker=lambda _a, _g: None,
            embedding_checker=lambda _a, _g: None,
        )
        result_explicit = run_export(
            store_explicit,
            _SPEC,
            plan,
            target="ollama",
            cached_base_dir=cached_base_explicit,
            subprocess_runner=_SubprocessRecorder(store_explicit.export_quant_dir(plan.quant)),
            vendor_override=vendor_explicit,
            ollama_create_runner=_fake_create,
            ollama_run_runner=_fake_run,
            vocab_checker=lambda _a, _g: None,
            embedding_checker=lambda _a, _g: None,
        )

        assert result_default.target == "ollama"
        assert result_explicit.target == "ollama"
        assert result_default.ollama_name == "mydoc:latest"
        assert result_explicit.ollama_name == "mydoc:latest"
        assert result_default.smoke_output_first_line == "hello from ollama"
        assert result_explicit.smoke_output_first_line == "hello from ollama"
        assert create_calls == [
            ("mydoc:latest", "Modelfile"),
            ("mydoc:latest", "Modelfile"),
        ]
        assert run_calls == ["mydoc:latest", "mydoc:latest"]
        assert _relative_file_bytes(result_default.export_dir) == _relative_file_bytes(
            result_explicit.export_dir
        )


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
            vocab_checker=lambda _a, _g: None,
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
            vocab_checker=lambda _a, _g: None,
        )
        assert r2.cached is True
        # Only the adapter conversion runs on the cached path.
        assert len(recorder2.commands) == 1
        assert any("convert_lora_to_gguf.py" in str(a) for a in recorder2.commands[0])

    def test_bad_cached_manifest_logs_warning_and_rebuilds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from dlm.export.errors import ExportManifestError
        from dlm.export.runner import _cached_base_matches

        export_dir = tmp_path / "exports" / "Q4_K_M"
        export_dir.mkdir(parents=True)
        base_gguf = export_dir / "base.Q4_K_M.gguf"
        base_gguf.write_bytes(b"cached bytes")
        (export_dir / "export_manifest.json").write_text("{}", encoding="utf-8")

        def _raise(_export_dir: Path) -> object:
            raise ExportManifestError("bad manifest")

        monkeypatch.setattr("dlm.export.manifest.load_export_manifest", _raise)
        caplog.set_level(logging.WARNING, logger="dlm.export.runner")

        assert _cached_base_matches(export_dir, base_gguf, "Q4_K_M") is False
        assert "export cache ignored stale manifest" in caplog.text


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

    def test_merged_export_delegates_to_merge_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cached_base, store, vendor = _setup_store(tmp_path, use_qlora=False)
        plan = ExportPlan(merged=True, dequantize_confirmed=True)
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))
        seen: list[dict[str, object]] = []

        def _fake_merge_path(**kwargs: object) -> None:
            seen.append(kwargs)

        monkeypatch.setattr("dlm.export.runner._perform_merge_path", _fake_merge_path)

        result = run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=recorder,
            vendor_override=vendor,
            skip_ollama=True,
            vocab_checker=lambda _a, _g: None,
            embedding_checker=lambda _a, _g: None,
        )

        assert result.merged is True
        assert len(seen) == 1
        assert seen[0]["adapter_path"] == store.resolve_current_adapter()
        assert seen[0]["was_qlora"] is False


class TestDefaultVocabCheck:
    """Default path loads the adapter tokenizer-vocab and compares against the base GGUF."""

    def test_mismatch_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from dlm.export.errors import PreflightError
        from dlm.export.runner import _default_check_base_vocab

        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "tokenizer_config.json").write_text(
            json.dumps({"vocab_size": 32000, "chat_template": "{{m}}"})
        )

        base = tmp_path / "base.gguf"
        base.write_bytes(b"ignored")

        # Force a specific GGUF vocab through the reader seam.
        monkeypatch.setattr("dlm.export.tokenizer_sync.read_gguf_vocab_size", lambda _p: 32001)
        with pytest.raises(PreflightError, match="gguf_vocab"):
            _default_check_base_vocab(adapter, base)

    def test_match_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from dlm.export.runner import _default_check_base_vocab

        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "tokenizer_config.json").write_text(
            json.dumps({"vocab_size": 32000, "chat_template": "{{m}}"})
        )
        base = tmp_path / "base.gguf"
        base.write_bytes(b"ignored")

        monkeypatch.setattr("dlm.export.tokenizer_sync.read_gguf_vocab_size", lambda _p: 32000)
        _default_check_base_vocab(adapter, base)


class TestVocabCheck:
    """`run_export` calls the vocab checker after base conversion (audit-04 B1)."""

    def test_mismatch_raises_preflight(self, tmp_path: Path) -> None:
        from dlm.export.errors import PreflightError

        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q4_K_M")
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))

        def _raise(_a: Path, _g: Path) -> None:
            raise PreflightError(probe="gguf_vocab", detail="adapter=32000 gguf=32001")

        with pytest.raises(PreflightError, match="gguf_vocab"):
            run_export(
                store,
                _SPEC,
                plan,
                cached_base_dir=cached_base,
                subprocess_runner=recorder,
                vendor_override=vendor,
                skip_ollama=True,
                vocab_checker=_raise,
            )

    def test_checker_receives_adapter_and_base_paths(self, tmp_path: Path) -> None:
        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q4_K_M")
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))
        seen: list[tuple[Path, Path]] = []

        def _record(adapter: Path, base: Path) -> None:
            seen.append((adapter, base))

        run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=recorder,
            vendor_override=vendor,
            skip_ollama=True,
            vocab_checker=_record,
        )
        assert len(seen) == 1
        adapter, base = seen[0]
        assert adapter == store.resolve_current_adapter()
        assert base.name == "base.Q4_K_M.gguf"


class TestEmbeddingCheck:
    """`run_export` invokes the embedding checker after the vocab checker (sprint 11.5)."""

    def test_mismatch_raises_preflight(self, tmp_path: Path) -> None:
        from dlm.export.errors import PreflightError

        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q4_K_M")
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))

        def _raise(_a: Path, _g: Path) -> None:
            raise PreflightError(probe="embedding_row_sha", detail="row 4 diverged")

        with pytest.raises(PreflightError, match="embedding_row_sha"):
            run_export(
                store,
                _SPEC,
                plan,
                cached_base_dir=cached_base,
                subprocess_runner=recorder,
                vendor_override=vendor,
                skip_ollama=True,
                vocab_checker=lambda _a, _g: None,
                embedding_checker=_raise,
            )

    def test_checker_receives_paths_after_vocab(self, tmp_path: Path) -> None:
        """Both checkers fire once per export, vocab before embedding."""
        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q4_K_M")
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))
        order: list[str] = []

        def _vocab(_a: Path, _g: Path) -> None:
            order.append("vocab")

        def _embedding(_a: Path, _g: Path) -> None:
            order.append("embedding")

        run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=recorder,
            vendor_override=vendor,
            skip_ollama=True,
            vocab_checker=_vocab,
            embedding_checker=_embedding,
        )
        assert order == ["vocab", "embedding"]


class TestImatrixWiring:
    """Sprint 11.6: runner routes plan.imatrix into the quantize pipeline."""

    def test_off_mode_passes_no_imatrix_flag(self, tmp_path: Path) -> None:
        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q4_K_M", imatrix="off")
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))
        run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=recorder,
            vendor_override=vendor,
            skip_ollama=True,
            vocab_checker=lambda _a, _g: None,
            embedding_checker=lambda _a, _g: None,
        )
        quantize_calls = [c for c in recorder.commands if c and "llama-quantize" in c[0]]
        assert len(quantize_calls) == 1
        assert "--imatrix" not in quantize_calls[0]

    def test_non_k_quant_skips_imatrix(self, tmp_path: Path) -> None:
        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q8_0", imatrix="auto")  # Q8_0 ignores imatrix
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))
        run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=recorder,
            vendor_override=vendor,
            skip_ollama=True,
            vocab_checker=lambda _a, _g: None,
            embedding_checker=lambda _a, _g: None,
        )
        # No llama-imatrix call in the recorder.
        assert not any("llama-imatrix" in c[0] for c in recorder.commands if c)

    def test_auto_with_empty_corpus_falls_back(self, tmp_path: Path) -> None:
        """Empty replay corpus → static quantize (no --imatrix flag)."""
        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q4_K_M", imatrix="auto")
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))
        # No replay corpus exists (store.ensure_layout doesn't create it).
        run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=recorder,
            vendor_override=vendor,
            skip_ollama=True,
            vocab_checker=lambda _a, _g: None,
            embedding_checker=lambda _a, _g: None,
        )
        quantize = [c for c in recorder.commands if c and "llama-quantize" in c[0]]
        assert len(quantize) == 1
        assert "--imatrix" not in quantize[0]

    def test_auto_with_corpus_threads_imatrix(self, tmp_path: Path) -> None:
        """When replay has content, the runner builds and passes --imatrix."""
        from datetime import UTC
        from datetime import datetime as _dt

        from dlm.replay.models import SectionSnapshot
        from dlm.replay.store import ReplayStore

        cached_base, store, vendor = _setup_store(tmp_path)
        # The base `_setup_store` only writes llama-quantize; imatrix
        # builds need a stub binary too.
        (vendor / "build" / "bin" / "llama-imatrix").write_text("# mock")

        # Seed the replay corpus with some text.
        ReplayStore.at(store.replay_corpus, store.replay_index).append_many(
            [
                SectionSnapshot(
                    section_id=f"{i:016x}",
                    section_type="prose",
                    content="Calibration text lorem ipsum " * 40,
                    first_seen_at=_dt(2026, 4, 19, tzinfo=UTC).replace(tzinfo=None),
                    last_seen_at=_dt(2026, 4, 19, tzinfo=UTC).replace(tzinfo=None),
                )
                for i in range(3)
            ]
        )

        plan = ExportPlan(quant="Q4_K_M", imatrix="auto")
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))

        # Wrap the recorder so imatrix subprocess calls also materialize
        # their `-o` file (lets build_imatrix's final existence check pass).
        def wrapped(cmd: Any) -> Any:
            cmd_list = list(cmd)
            if cmd_list and "llama-imatrix" in cmd_list[0]:
                recorder.commands.append(cmd_list)
                out_ix = cmd_list.index("-o") + 1
                Path(cmd_list[out_ix]).write_bytes(b"fake imatrix bytes")
                return None
            return recorder(cmd)

        run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=wrapped,
            vendor_override=vendor,
            skip_ollama=True,
            vocab_checker=lambda _a, _g: None,
            embedding_checker=lambda _a, _g: None,
        )

        imatrix_calls = [c for c in recorder.commands if c and "llama-imatrix" in c[0]]
        assert len(imatrix_calls) == 1
        quantize = [c for c in recorder.commands if c and "llama-quantize" in c[0]]
        assert len(quantize) == 1
        assert "--imatrix" in quantize[0]


class TestDraftWiring:
    """Sprint 12.5: runner routes draft override/disable through to the Modelfile."""

    def test_override_tag_written_to_modelfile(self, tmp_path: Path) -> None:
        """--draft custom:tag propagates to PARAMETER draft_model."""
        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q4_K_M", ollama_name="mydoc")
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))

        # skip_ollama=True stops before ollama_create but *after* the
        # Modelfile is rendered? Actually no — the Modelfile is only
        # rendered when skip_ollama=False. Switch to a mock ollama_create.
        def _ollama_create(*, name: str, modelfile_path: Path, cwd: Path) -> str:
            _ = name, cwd
            assert modelfile_path.is_file()
            text = modelfile_path.read_text()
            assert "PARAMETER draft_model custom:tag" in text
            return "created"

        run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=recorder,
            vendor_override=vendor,
            skip_ollama=False,
            skip_smoke=True,
            vocab_checker=lambda _a, _g: None,
            embedding_checker=lambda _a, _g: None,
            ollama_create_runner=_ollama_create,
            draft_override="custom:tag",
        )

    def test_no_draft_suppresses_emission(self, tmp_path: Path) -> None:
        cached_base, store, vendor = _setup_store(tmp_path)
        plan = ExportPlan(quant="Q4_K_M", ollama_name="mydoc")
        recorder = _SubprocessRecorder(store.export_quant_dir(plan.quant))

        def _ollama_create(*, name: str, modelfile_path: Path, cwd: Path) -> str:
            _ = name, cwd
            assert "PARAMETER draft_model" not in modelfile_path.read_text()
            return "created"

        run_export(
            store,
            _SPEC,
            plan,
            cached_base_dir=cached_base,
            subprocess_runner=recorder,
            vendor_override=vendor,
            skip_ollama=False,
            skip_smoke=True,
            vocab_checker=lambda _a, _g: None,
            embedding_checker=lambda _a, _g: None,
            ollama_create_runner=_ollama_create,
            draft_disabled=True,
        )


class TestDefaultEmbeddingCheck:
    """Exercise `_default_check_embedding_rows`'s skip path (audit-04 Q2)."""

    def test_no_modules_to_save_is_noop(self, tmp_path: Path) -> None:
        """Default path skips silently when the adapter has no modules_to_save."""
        from dlm.export.runner import _default_check_embedding_rows

        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "x", "peft_type": "LORA"})
        )
        (adapter / "tokenizer_config.json").write_text(
            json.dumps({"vocab_size": 5, "chat_template": "x"})
        )
        base = tmp_path / "base.gguf"
        base.write_bytes(b"would-not-parse")
        # Skip path: no raise even though the GGUF is nonsense.
        _default_check_embedding_rows(adapter, base)


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

    def test_missing_adapter_override_raises(self, tmp_path: Path) -> None:
        cached_base, store, vendor = _setup_store(tmp_path)

        with pytest.raises(ExportError, match="adapter_path_override .* does not exist"):
            run_export(
                store,
                _SPEC,
                ExportPlan(),
                cached_base_dir=cached_base,
                subprocess_runner=lambda _cmd: None,
                vendor_override=vendor,
                adapter_path_override=tmp_path / "missing",
            )

    def test_missing_named_adapter_raises(self, tmp_path: Path) -> None:
        cached_base, store, vendor = _setup_store(tmp_path)

        with pytest.raises(ExportError, match="run `dlm train` before exporting for adapter"):
            run_export(
                store,
                _SPEC,
                ExportPlan(),
                cached_base_dir=cached_base,
                subprocess_runner=lambda _cmd: None,
                vendor_override=vendor,
                adapter_name="knowledge",
            )

    def test_named_adapter_export_uses_named_pointer(self, tmp_path: Path) -> None:
        cached_base, store, vendor = _setup_named_store(tmp_path)
        recorder = _SubprocessRecorder(store.export_quant_dir("Q4_K_M"))

        result = run_export(
            store,
            _SPEC,
            ExportPlan(),
            cached_base_dir=cached_base,
            subprocess_runner=recorder,
            vendor_override=vendor,
            skip_ollama=True,
            vocab_checker=lambda _a, _g: None,
            embedding_checker=lambda _a, _g: None,
            adapter_name="knowledge",
        )

        assert result.export_dir == store.export_quant_dir("Q4_K_M")
        assert len(recorder.commands) == 3


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
            vocab_checker=lambda _a, _g: None,
        )

        manifest = load_manifest(store.manifest)
        assert len(manifest.exports) == 1
        export = manifest.exports[0]
        assert export.target == "ollama"
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
                vocab_checker=lambda _a, _g: None,
            )

        # Peer released → no export summary landed (we errored before save).
        manifest = load_manifest(store.manifest)
        assert len(manifest.exports) == 0


class TestRunnerInternals:
    def test_cached_base_missing_manifest_is_false(self, tmp_path: Path) -> None:
        from dlm.export.runner import _cached_base_matches

        export_dir = tmp_path / "exports" / "Q4_K_M"
        export_dir.mkdir(parents=True)
        base_gguf = export_dir / "base.Q4_K_M.gguf"
        base_gguf.write_bytes(b"cached bytes")

        assert _cached_base_matches(export_dir, base_gguf, "Q4_K_M") is False

    def test_cached_base_quant_mismatch_is_false(self, tmp_path: Path) -> None:
        from dlm.export.manifest import ExportManifest
        from dlm.export.runner import _cached_base_matches

        export_dir = tmp_path / "exports" / "Q4_K_M"
        export_dir.mkdir(parents=True)
        base_gguf = export_dir / "base.Q4_K_M.gguf"
        base_gguf.write_bytes(b"cached bytes")
        manifest = ExportManifest(
            target="ollama",
            quant="Q5_K_M",
            created_at=datetime(2026, 4, 23, 12, 0, 0),
            created_by="dlm-test",
            base_model_hf_id="org/base",
            base_model_revision="a" * 40,
            adapter_version=1,
            artifacts=[],
        )
        (export_dir / "export_manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        assert _cached_base_matches(export_dir, base_gguf, "Q4_K_M") is False

    def test_cached_base_without_recorded_artifact_is_false(self, tmp_path: Path) -> None:
        from dlm.export.manifest import ExportManifest, build_artifact
        from dlm.export.runner import _cached_base_matches

        export_dir = tmp_path / "exports" / "Q4_K_M"
        export_dir.mkdir(parents=True)
        base_gguf = export_dir / "base.Q4_K_M.gguf"
        other = export_dir / "other.gguf"
        base_gguf.write_bytes(b"cached bytes")
        other.write_bytes(b"other bytes")
        manifest = ExportManifest(
            target="ollama",
            quant="Q4_K_M",
            created_at=datetime(2026, 4, 23, 12, 0, 0),
            created_by="dlm-test",
            base_model_hf_id="org/base",
            base_model_revision="a" * 40,
            adapter_version=1,
            artifacts=[build_artifact(export_dir, other)],
        )
        (export_dir / "export_manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        assert _cached_base_matches(export_dir, base_gguf, "Q4_K_M") is False

    def test_cached_imatrix_without_existing_file_returns_none(self, tmp_path: Path) -> None:
        from dlm.export.runner import _resolve_or_build_imatrix

        cached_base, store, _vendor = _setup_store(tmp_path)
        fp16 = tmp_path / "base.fp16.gguf"
        fp16.write_bytes(b"fp16")

        assert (
            _resolve_or_build_imatrix(
                export_dir=tmp_path,
                fp16_path=fp16,
                plan=ExportPlan(quant="Q4_K_M", imatrix="cached"),
                run=lambda _cmd: None,
                vendor_override=None,
                spec=_SPEC,
                store=store,
            )
            is None
        )

    def test_auto_imatrix_cache_hit_logs_and_returns_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from types import SimpleNamespace

        from dlm.export.runner import _resolve_or_build_imatrix

        cached_base, store, _vendor = _setup_store(tmp_path)
        fp16 = tmp_path / "base.fp16.gguf"
        fp16.write_bytes(b"fp16")
        imatrix = tmp_path / "imatrix.gguf"
        imatrix.write_bytes(b"imatrix")

        monkeypatch.setattr(
            "dlm.export.imatrix.calibration_text_from_replay",
            lambda **_kwargs: ("calibration text", "abc123"),
        )
        monkeypatch.setattr(
            "dlm.export.imatrix.resolve_imatrix",
            lambda *_args, **_kwargs: SimpleNamespace(path=imatrix, sha256="abcdef123456"),
        )
        caplog.set_level(logging.INFO, logger="dlm.export.runner")

        resolved = _resolve_or_build_imatrix(
            export_dir=tmp_path,
            fp16_path=fp16,
            plan=ExportPlan(quant="Q4_K_M", imatrix="auto"),
            run=lambda _cmd: None,
            vendor_override=None,
            spec=_SPEC,
            store=store,
        )

        assert resolved == imatrix
        assert "imatrix: cache hit (" in caplog.text

    def test_run_ollama_stage_records_detected_version(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from dlm.export.runner import _run_ollama_stage

        cached_base, store, _vendor = _setup_store(tmp_path)
        export_dir = store.export_quant_dir("Q4_K_M")
        export_dir.mkdir(parents=True, exist_ok=True)
        base_gguf = export_dir / "base.Q4_K_M.gguf"
        base_gguf.write_bytes(b"base")
        adapter = store.resolve_current_adapter()
        assert adapter is not None

        monkeypatch.setattr("dlm.export.ollama.check_ollama_version", lambda: (1, 2, 3))
        monkeypatch.setattr("dlm.export.draft_registry.resolve_draft", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            "dlm.export.ollama.render_modelfile",
            lambda _ctx: "FROM ./base.Q4_K_M.gguf\n",
        )

        seen: list[str] = []

        def _create(*, name: str, modelfile_path: Path, cwd: Path) -> str:
            seen.append(name)
            assert modelfile_path.exists()
            assert cwd == export_dir
            return "created"

        monkeypatch.setattr("dlm.export.ollama.ollama_create", _create)
        monkeypatch.setattr("dlm.export.ollama.ollama_run", lambda **_kwargs: "unused")

        modelfile_path, name, ver_str, smoke_first_line = _run_ollama_stage(
            store=store,
            spec=_SPEC,
            plan=ExportPlan(quant="Q4_K_M"),
            adapter_path=adapter,
            export_dir=export_dir,
            base_gguf_path=base_gguf,
            adapter_version=1,
            system_prompt=None,
            source_dlm_path=None,
            skip_smoke=True,
            ollama_create_runner=None,
            ollama_run_runner=None,
            training_sequence_len=None,
            override_temperature=None,
            override_top_p=None,
            draft_override=None,
            draft_disabled=False,
        )

        assert modelfile_path.exists()
        assert name == seen[0]
        assert ver_str == "1.2.3"
        assert smoke_first_line is None
