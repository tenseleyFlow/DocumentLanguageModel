"""End-to-end GGUF export on the SmolLM2-135M fixture.

Sprint 11 DoD: produce a valid GGUF file readable by `llama-cli`, with
LoRA A/B tensors referencing the correct base tensor names.

Marked `@pytest.mark.slow`. Requires:
- `vendor/llama.cpp/` submodule initialized and built (`scripts/bump-llama-cpp.sh build`)
- SmolLM2-135M offline cache (from Sprint 02's fixture)
- A prior `dlm train` run against that base to produce an adapter

When any dependency is missing the test skips with a clear message.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.slow
def test_export_produces_valid_gguf() -> None:
    """Full `dlm export` cycle on the tiny model.

    Shape:
      1. `vendor/llama.cpp/build/bin/llama-quantize` exists → else skip.
      2. SmolLM2-135M fixture resolvable → else skip.
      3. `dlm train` produces an adapter in a fresh tmp store → else skip.
      4. `run_export(store, spec, plan=Q4_K_M)` emits base + adapter GGUF.
      5. `llama-cli -m base.Q4_K_M.gguf --lora adapter.gguf -p "..."` returns
         non-empty stdout.
    """
    vendor_root = Path(__file__).resolve().parents[3] / "vendor" / "llama.cpp"
    if not (vendor_root / "build" / "bin" / "llama-quantize").is_file():
        pytest.skip("vendor/llama.cpp not built; run `scripts/bump-llama-cpp.sh build` to enable.")

    try:
        from tests.fixtures.tiny_model import tiny_model_path

        tiny_model_path()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tiny-model fixture unavailable: {exc}")

    pytest.xfail("export integration scaffolded; body deferred to first CI slow run")


@pytest.mark.slow
def test_qlora_merge_requires_dequantize_flag() -> None:
    """Contract: `--merged` on a QLoRA adapter without `--dequantize` refuses.

    Handled entirely in the plan's safety gate; unit-tested at
    `tests/unit/export/test_plan.py::TestMergeSafetyGate`. This
    integration test re-asserts it survives the full CLI path so a
    future refactor doesn't silently remove the guardrail.
    """
    pytest.xfail("CLI integration scaffolded; body deferred")
