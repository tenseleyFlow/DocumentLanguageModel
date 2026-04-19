"""Two simultaneous `run_export` calls must serialize cleanly (audit-04 T2).

The store exclusive lock (audit-04 B3) is the contract that prevents
`manifest.exports` from dropping a row when two quants land at once.
This test runs two in-process threads pointed at the same store with
different quants; both must succeed and the manifest must end up with
two rows.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dlm.base_models import BASE_MODELS
from dlm.export import ExportPlan, run_export
from dlm.store.manifest import Manifest, load_manifest, save_manifest
from dlm.store.paths import for_dlm

_SPEC = BASE_MODELS["smollm2-135m"]


class _Recorder:
    """Fake subprocess runner that writes the files the pipeline expects."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd: Any) -> Any:
        cmd_list = list(cmd)
        self.calls.append(cmd_list)

        if any(a.endswith("convert_hf_to_gguf.py") for a in cmd_list):
            for i, a in enumerate(cmd_list):
                if a == "--outfile" and i + 1 < len(cmd_list):
                    Path(cmd_list[i + 1]).write_bytes(b"fake-hf")
        if cmd_list and ("llama-quantize" in cmd_list[0] or cmd_list[0].endswith("quantize")):
            Path(cmd_list[-2]).write_bytes(b"fake-quant")
        if any(a.endswith("convert_lora_to_gguf.py") for a in cmd_list):
            for i, a in enumerate(cmd_list):
                if a == "--outfile" and i + 1 < len(cmd_list):
                    Path(cmd_list[i + 1]).write_bytes(b"fake-lora")
        return None


def _setup_store(tmp_path: Path) -> tuple[Path, Any, Path]:
    store = for_dlm("01CONCUR", home=tmp_path)
    store.ensure_layout()
    save_manifest(store.manifest, Manifest(dlm_id="01CONCUR", base_model=_SPEC.key))

    adapter = store.adapter_version(1)
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": _SPEC.hf_id, "peft_type": "LORA"})
    )
    (adapter / "tokenizer_config.json").write_text(
        json.dumps({"vocab_size": 32000, "chat_template": "{{m}}"})
    )
    (adapter / "training_run.json").write_text(json.dumps({"use_qlora": False}))
    store.set_current_adapter(adapter)

    cached_base = tmp_path / "cache" / "base"
    cached_base.mkdir(parents=True)

    vendor = tmp_path / "vendor" / "llama.cpp"
    vendor.mkdir(parents=True)
    (vendor / "convert_hf_to_gguf.py").write_text("# mock")
    (vendor / "convert_lora_to_gguf.py").write_text("# mock")
    (vendor / "build" / "bin").mkdir(parents=True)
    (vendor / "build" / "bin" / "llama-quantize").write_text("# mock")
    (vendor / "VERSION").write_text("b9999\n")
    return cached_base, store, vendor


def _export(quant: str, *, store: Any, cached_base: Path, vendor: Path) -> str:
    plan = ExportPlan(quant=quant, ollama_name=f"race-{quant.lower()}")
    run_export(
        store,
        _SPEC,
        plan,
        cached_base_dir=cached_base,
        subprocess_runner=_Recorder(),
        vendor_override=vendor,
        skip_ollama=True,
        vocab_checker=lambda _a, _g: None,
    )
    return quant


def test_concurrent_exports_both_land(tmp_path: Path) -> None:
    """Two quants of the same adapter → manifest has two export rows."""
    cached_base, store, vendor = _setup_store(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_export, q, store=store, cached_base=cached_base, vendor=vendor)
            for q in ("Q4_K_M", "Q5_K_M")
        ]
        results = {f.result() for f in as_completed(futures)}

    assert results == {"Q4_K_M", "Q5_K_M"}

    manifest = load_manifest(store.manifest)
    assert len(manifest.exports) == 2
    assert {e.quant for e in manifest.exports} == {"Q4_K_M", "Q5_K_M"}
