"""Unit tests for the single-file VL GGUF emitter.

Covers the three refusal branches on `_assert_supported` (verdict not
SUPPORTED, plan not merged, imatrix not "off"), the happy path (merge
runner + subprocess runner as injection seams, manifest + Modelfile +
sidecar all land on disk), and the adapter-resolution typed refusal.

The emitter shells out to `convert_hf_to_gguf.py` and `llama-quantize`;
unit tests substitute a recorder for the subprocess runner so we can
assert argv shape without needing a vendored tree. `merge_runner` is
similarly substituted so the tests don't need a real PEFT model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import dlm.export.vl_gguf as vl_gguf
from dlm.base_models.schema import BaseModelSpec, VlPreprocessorPlan
from dlm.export.arch_probe import ArchProbeResult, SupportLevel
from dlm.export.errors import ExportError, VlGgufUnsupportedError
from dlm.export.plan import ExportPlan
from dlm.export.vl_gguf import VlGgufResult, run_vl_gguf_export
from dlm.store.paths import for_dlm

_VALID_ULID = "01KPMGSTNGSTTSTTSTTSTTSTVA"


def _qwen2vl_spec() -> BaseModelSpec:
    return BaseModelSpec(
        key="qwen2-vl-2b",
        hf_id="Qwen/Qwen2-VL-2B-Instruct",
        revision="c" * 40,
        architecture="Qwen2VLForConditionalGeneration",
        params=2_000_000_000,
        target_modules=["q_proj", "v_proj"],
        template="qwen2-vl",
        gguf_arch="qwen2vl",
        tokenizer_pre="qwen2",
        license_spdx="Apache-2.0",
        redistributable=True,
        size_gb_fp16=4.0,
        context_length=32768,
        recommended_seq_len=2048,
        modality="vision-language",
        vl_preprocessor_plan=VlPreprocessorPlan(
            target_size=(448, 448),
            image_token="<|image_pad|>",
            num_image_tokens=256,
        ),
    )


def _supported_verdict() -> ArchProbeResult:
    return ArchProbeResult(
        arch_class="Qwen2VLForConditionalGeneration",
        support=SupportLevel.SUPPORTED,
        reason="arch registered on TextModel subclass",
        llama_cpp_tag="b8816",
    )


def _fixture_llama_cpp_root(tmp_path: Path) -> Path:
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("# fake converter\n", encoding="utf-8")
    (root / "build" / "bin" / "llama-quantize").write_text("# fake quantize\n", encoding="utf-8")
    return root


def _unsupported_verdict() -> ArchProbeResult:
    return ArchProbeResult(
        arch_class="PaliGemmaForConditionalGeneration",
        support=SupportLevel.UNSUPPORTED,
        reason="arch absent from convert_hf_to_gguf.py",
        llama_cpp_tag="b8816",
    )


def _partial_verdict() -> ArchProbeResult:
    return ArchProbeResult(
        arch_class="InternVL2",
        support=SupportLevel.PARTIAL,
        reason="arch only on MmprojModel subclass",
        llama_cpp_tag="b8816",
    )


def _merged_plan(**overrides: object) -> ExportPlan:
    kwargs: dict[str, object] = {
        "merged": True,
        "imatrix": "off",
        "quant": "Q4_K_M",
    }
    kwargs.update(overrides)
    return ExportPlan(**kwargs)  # type: ignore[arg-type]


def _populate_adapter(store: Any, version: int = 1) -> Path:
    """Write a minimally-valid adapter checkpoint under v0001."""
    store.ensure_layout()
    adapter: Path = store.adapter_version(version)
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
                "target_modules": ["q_proj", "v_proj"],
                "r": 16,
            }
        ),
        encoding="utf-8",
    )
    # Chat template + vocab_size satisfies preflight.
    (adapter / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "vocab_size": 151643,
                "chat_template": "{{ 'hi' }}",
            }
        ),
        encoding="utf-8",
    )
    (adapter / "training_run.json").write_text(
        json.dumps({"use_qlora": False}),
        encoding="utf-8",
    )
    store.set_current_adapter(adapter)
    return adapter


def _populate_named_adapter(store: Any, name: str, version: int = 1) -> Path:
    """Write a minimally-valid named adapter checkpoint under `adapter/<name>/`."""
    store.ensure_layout()
    adapter: Path = store.adapter_version_for(name, version)
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
                "target_modules": ["q_proj", "v_proj"],
                "r": 16,
            }
        ),
        encoding="utf-8",
    )
    (adapter / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "vocab_size": 151643,
                "chat_template": "{{ 'hi' }}",
            }
        ),
        encoding="utf-8",
    )
    (adapter / "training_run.json").write_text(
        json.dumps({"use_qlora": False}),
        encoding="utf-8",
    )
    store.set_current_adapter_for(name, adapter)
    return adapter


class TestRefusals:
    """Covers `_assert_supported` — the three preconditions + adapter gate."""

    def test_unsupported_verdict_refused(self, tmp_path: Path) -> None:
        store = for_dlm(_VALID_ULID, home=tmp_path)
        _populate_adapter(store)
        with pytest.raises(VlGgufUnsupportedError, match="UNSUPPORTED"):
            run_vl_gguf_export(
                store,
                _qwen2vl_spec(),
                _merged_plan(),
                verdict=_unsupported_verdict(),
                cached_base_dir=tmp_path / "base",
                subprocess_runner=lambda _args: None,
                merge_runner=lambda *_a, **_kw: None,
            )

    def test_partial_verdict_refused(self, tmp_path: Path) -> None:
        store = for_dlm(_VALID_ULID, home=tmp_path)
        _populate_adapter(store)
        with pytest.raises(VlGgufUnsupportedError, match="PARTIAL"):
            run_vl_gguf_export(
                store,
                _qwen2vl_spec(),
                _merged_plan(),
                verdict=_partial_verdict(),
                cached_base_dir=tmp_path / "base",
                subprocess_runner=lambda _args: None,
                merge_runner=lambda *_a, **_kw: None,
            )

    def test_non_merged_plan_refused(self, tmp_path: Path) -> None:
        """VL GGUF is merged-only at the pinned tag — non-merged must refuse."""
        store = for_dlm(_VALID_ULID, home=tmp_path)
        _populate_adapter(store)
        with pytest.raises(VlGgufUnsupportedError, match="merged-only"):
            run_vl_gguf_export(
                store,
                _qwen2vl_spec(),
                ExportPlan(merged=False, imatrix="off", quant="Q4_K_M"),
                verdict=_supported_verdict(),
                cached_base_dir=tmp_path / "base",
                subprocess_runner=lambda _args: None,
                merge_runner=lambda *_a, **_kw: None,
            )

    def test_imatrix_refused(self, tmp_path: Path) -> None:
        """Replay is text-only → imatrix must be off for VL GGUF."""
        store = for_dlm(_VALID_ULID, home=tmp_path)
        _populate_adapter(store)
        with pytest.raises(VlGgufUnsupportedError, match="imatrix"):
            run_vl_gguf_export(
                store,
                _qwen2vl_spec(),
                _merged_plan(imatrix="auto"),
                verdict=_supported_verdict(),
                cached_base_dir=tmp_path / "base",
                subprocess_runner=lambda _args: None,
                merge_runner=lambda *_a, **_kw: None,
            )

    def test_missing_adapter_refused(self, tmp_path: Path) -> None:
        """Empty store → `VlGgufUnsupportedError` (catches via dispatcher)."""
        store = for_dlm(_VALID_ULID, home=tmp_path)
        store.ensure_layout()
        with pytest.raises(VlGgufUnsupportedError, match="no current adapter"):
            run_vl_gguf_export(
                store,
                _qwen2vl_spec(),
                _merged_plan(),
                verdict=_supported_verdict(),
                cached_base_dir=tmp_path / "base",
                subprocess_runner=lambda _args: None,
                merge_runner=lambda *_a, **_kw: None,
            )


class TestHappyPath:
    """Verifies the emitter's argv sequence + on-disk artifacts."""

    def _run(
        self,
        tmp_path: Path,
        *,
        plan: ExportPlan | None = None,
    ) -> tuple[VlGgufResult, list[list[str]]]:
        store = for_dlm(_VALID_ULID, home=tmp_path)
        adapter_dir = _populate_adapter(store)
        cached_base = tmp_path / "base-cache"
        cached_base.mkdir()
        llama_cpp_root = _fixture_llama_cpp_root(tmp_path)

        recorded_argv: list[list[str]] = []

        def _recorder(args: Any) -> None:
            # The real runner writes the output file as a side effect;
            # we stub that here so the post-quantize existence check
            # passes without a real llama-quantize binary.
            recorded_argv.append(list(args))
            if args and args[0].endswith("llama-quantize"):
                # args shape: [binary, in_fp16, out_quant, quant]
                Path(args[2]).write_bytes(b"stub-gguf-bytes")

        merge_calls: list[tuple[Path, Path, Path]] = []

        def _merge(adapter: Path, out_dir: Path, *, cached_base_dir: Path) -> None:
            merge_calls.append((adapter, out_dir, cached_base_dir))
            out_dir.mkdir(parents=True, exist_ok=True)

        result = run_vl_gguf_export(
            store,
            _qwen2vl_spec(),
            plan or _merged_plan(),
            verdict=_supported_verdict(),
            cached_base_dir=cached_base,
            subprocess_runner=_recorder,
            merge_runner=_merge,
            dlm_version="test",
            llama_cpp_root_override=llama_cpp_root,
        )
        # Merge is invoked exactly once, against the current adapter.
        assert len(merge_calls) == 1
        assert merge_calls[0][0] == adapter_dir
        return result, recorded_argv

    def test_argv_shape_convert_then_quantize(self, tmp_path: Path) -> None:
        _, argv = self._run(tmp_path)
        assert len(argv) == 2, f"expected convert + quantize, got {argv}"
        convert_cmd, quantize_cmd = argv
        # convert_hf_to_gguf.py is the python-script invocation.
        assert convert_cmd[1].endswith("convert_hf_to_gguf.py")
        assert "--outfile" in convert_cmd
        assert "--outtype" in convert_cmd
        # llama-quantize is the binary: [bin, in, out, quant]
        assert quantize_cmd[0].endswith("llama-quantize")
        assert quantize_cmd[-1] == "Q4_K_M"

    def test_writes_modelfile_and_manifest(self, tmp_path: Path) -> None:
        result, _ = self._run(tmp_path)
        assert result.modelfile_path.exists()
        assert result.manifest_path.exists()
        body = result.modelfile_path.read_text(encoding="utf-8")
        assert "FROM ./base.Q4_K_M.gguf" in body
        # Merged-only: no ADAPTER directive.
        assert "ADAPTER" not in body
        # Single-file contract: mmproj_path is None.
        assert result.mmproj_path is None

    def test_sidecar_records_verdict_and_sha(self, tmp_path: Path) -> None:
        result, _ = self._run(tmp_path)
        sidecar_path = result.export_dir / "vl_gguf.json"
        assert sidecar_path.exists()
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["schema"] == 1
        assert sidecar["mmproj_path"] is None
        assert sidecar["arch_verdict"]["support"] == "SUPPORTED"
        assert sidecar["arch_verdict"]["architecture"] == "Qwen2VLForConditionalGeneration"
        assert sidecar["arch_verdict"]["llama_cpp_tag"] == "b8816"
        # sha256 is a 64-char hex digest.
        assert len(sidecar["gguf_sha256"]) == 64

    def test_manifest_records_tag_and_artifacts(self, tmp_path: Path) -> None:
        result, _ = self._run(tmp_path)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["llama_cpp_tag"] == "b8816"
        assert manifest["merged"] is True
        assert manifest["quant"] == "Q4_K_M"
        paths = {a["path"] for a in manifest["artifacts"]}
        assert "base.Q4_K_M.gguf" in paths
        assert "Modelfile" in paths

    def test_named_adapter_export_uses_named_current_pointer(self, tmp_path: Path) -> None:
        store = for_dlm(_VALID_ULID, home=tmp_path)
        flat = _populate_adapter(store, version=1)
        named = _populate_named_adapter(store, "knowledge", version=2)
        cached_base = tmp_path / "base-cache"
        cached_base.mkdir()
        llama_cpp_root = _fixture_llama_cpp_root(tmp_path)

        merge_calls: list[tuple[Path, Path, Path]] = []

        def _recorder(args: Any) -> None:
            if args and args[0].endswith("llama-quantize"):
                Path(args[2]).write_bytes(b"stub-gguf-bytes")

        def _merge(adapter: Path, out_dir: Path, *, cached_base_dir: Path) -> None:
            merge_calls.append((adapter, out_dir, cached_base_dir))
            out_dir.mkdir(parents=True, exist_ok=True)

        result = run_vl_gguf_export(
            store,
            _qwen2vl_spec(),
            _merged_plan(),
            verdict=_supported_verdict(),
            cached_base_dir=cached_base,
            adapter_name="knowledge",
            subprocess_runner=_recorder,
            merge_runner=_merge,
            llama_cpp_root_override=llama_cpp_root,
        )

        assert len(merge_calls) == 1
        assert merge_calls[0][0] == named
        assert merge_calls[0][0] != flat
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["adapter_version"] == 2

    def test_missing_quantize_output_raises(self, tmp_path: Path) -> None:
        store = for_dlm(_VALID_ULID, home=tmp_path)
        _populate_adapter(store)
        cached_base = tmp_path / "base-cache"
        cached_base.mkdir()
        llama_cpp_root = _fixture_llama_cpp_root(tmp_path)

        def _recorder(_args: Any) -> None:
            return None

        def _merge(_adapter: Path, out_dir: Path, *, cached_base_dir: Path) -> None:
            out_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(ExportError, match="expected .*base.Q4_K_M.gguf"):
            run_vl_gguf_export(
                store,
                _qwen2vl_spec(),
                _merged_plan(),
                verdict=_supported_verdict(),
                cached_base_dir=cached_base,
                subprocess_runner=_recorder,
                merge_runner=_merge,
                llama_cpp_root_override=llama_cpp_root,
            )


class TestHelpers:
    def test_version_parser_falls_back_to_one_for_non_version_dir(self, tmp_path: Path) -> None:
        assert vl_gguf._version_from_dir_name(tmp_path / "merged-adapter") == 1

    def test_default_runner_delegates_to_run_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded: dict[str, object] = {}

        def _fake_run_checked(args: list[str], *, timeout: int) -> object:
            recorded["args"] = args
            recorded["timeout"] = timeout
            return "ok"

        monkeypatch.setattr(vl_gguf, "run_checked", _fake_run_checked)
        out = vl_gguf._default_runner(("python", "tool.py"))
        assert out == "ok"
        assert recorded == {"args": ["python", "tool.py"], "timeout": 60 * 60}
