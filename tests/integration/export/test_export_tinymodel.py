"""End-to-end GGUF export on the SmolLM2-135M fixture (Sprint 11).

Replaces the audit-04 M4 scaffold with a real body driven by the shared
`trained_store` fixture. Exercises the full `dlm export` CLI:

  dlm train smollm2-135m (via fixture) → dlm export --quant Q4_K_M --skip-ollama
    → base.Q4_K_M.gguf + adapter.gguf + export_manifest.json

Assertions:
- Base + adapter GGUF files exist with correct magic bytes.
- Vocab parity — `assert_gguf_vocab_matches` confirms the tokenizer the
  adapter was trained with lines up with the quantized base. This is
  the audit-04 B1 contract that prevents a silent tokenizer / GGUF
  vocab mismatch slipping through to `ollama run`.

Skips when:
- `vendor/llama.cpp/build/bin/llama-quantize` is missing (submodule
  not built). The integration-slow CI job builds it up-front.
- The `trained_store` fixture couldn't produce an adapter (transitive
  skip — see that fixture's own skip conditions).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytestmark = pytest.mark.slow


def _vendor_built() -> bool:
    vendor_root = Path(__file__).resolve().parents[3] / "vendor" / "llama.cpp"
    return (vendor_root / "build" / "bin" / "llama-quantize").is_file()


@pytest.mark.slow
def test_export_produces_valid_gguf(trained_store) -> None:
    if not _vendor_built():
        pytest.skip("vendor/llama.cpp not built; run `scripts/bump-llama-cpp.sh build`.")

    from dlm.cli.app import app
    from dlm.export.tokenizer_sync import assert_gguf_vocab_matches, tokenizer_from_adapter

    # `dlm export` reads DLM_HOME at invocation time; the trained_store
    # fixture set it, but re-assert here for clarity.
    os.environ["DLM_HOME"] = str(trained_store.home)

    # Clear offline env so `downloader.download_spec` can resolve the
    # already-cached base snapshot (the fixture warmed the cache; no new
    # download happens).
    offline_vars = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    saved = {k: os.environ.pop(k, None) for k in offline_vars}

    try:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "export",
                str(trained_store.doc),
                "--quant",
                "Q4_K_M",
                "--skip-ollama",
                "--no-smoke",
            ],
        )
        assert result.exit_code == 0, result.output

        export_dir = trained_store.store.export_quant_dir("Q4_K_M")
        base_gguf = export_dir / "base.Q4_K_M.gguf"
        adapter_gguf = export_dir / "adapter.gguf"

        assert base_gguf.is_file(), f"missing {base_gguf}"
        assert adapter_gguf.is_file(), f"missing {adapter_gguf}"
        assert base_gguf.read_bytes()[:4] == b"GGUF"
        assert adapter_gguf.read_bytes()[:4] == b"GGUF"

        # Vocab parity — audit-04 B1 contract.
        adapter_dir = trained_store.store.resolve_current_adapter()
        assert adapter_dir is not None
        tokenizer = tokenizer_from_adapter(adapter_dir)
        assert_gguf_vocab_matches(base_gguf, tokenizer)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


@pytest.mark.slow
def test_qlora_merge_requires_dequantize_flag(tmp_path: Path) -> None:
    """Contract: `--merged` on a QLoRA adapter without `--dequantize` refuses.

    Enforced at plan-resolve time; unit-tested at
    `tests/unit/export/test_plan.py::TestMergeSafetyGate`. This integration
    test re-asserts the refusal survives the full CLI path so a future
    refactor can't silently remove the guardrail.
    """
    from dlm.cli.app import app

    # Scaffold a fresh .dlm (no training needed — the gate fires before
    # the export runner touches the adapter).
    home = tmp_path / "dlm-home"
    home.mkdir()
    os.environ["DLM_HOME"] = str(home)

    from tests.fixtures.dlm_factory import make_dlm

    doc = home / "merge-gate.dlm"
    doc.write_text(make_dlm(base_model="smollm2-135m"), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            str(doc),
            "--quant",
            "Q4_K_M",
            "--merged",
            "--skip-ollama",
        ],
    )
    # The plan's merge-safety gate refuses at exit code 1 with a message
    # naming `--dequantize`. Unit tests check the wording; here we just
    # verify the CLI fails non-zero.
    assert result.exit_code != 0, (
        "merge gate regressed: CLI accepted --merged without --dequantize "
        f"on what was a QLoRA-capable flow. Output:\n{result.output}"
    )
