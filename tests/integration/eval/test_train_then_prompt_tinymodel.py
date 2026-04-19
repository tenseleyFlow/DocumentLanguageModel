"""End-to-end: train on tiny model, then `dlm prompt` against the adapter.

Sprint 10 DoD: `dlm prompt` one-shot works on a freshly trained tiny-model
adapter. Also exercises the cross-hardware `InferencePlan` path when run
on a CPU-only runner after a QLoRA adapter was CI-produced on a CUDA
job (the regression test that F05 calls out).

Marked `@pytest.mark.slow`. Skipped when the SmolLM2-135M fixture isn't
offline-resolvable (same gate as Sprint 09's integration stubs).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.slow
def test_train_then_prompt_one_cycle() -> None:
    """20-step train + prompt generates non-empty coherent output.

    Shape:
      1. Synthetic `.dlm` via `tests.fixtures.dlm_factory`.
      2. `trainer.run(..., max_steps=20)` on SmolLM2-135M.
      3. Resolve `InferencePlan` against current host's caps.
      4. `load_for_inference` → `generate(prompt="What is X?")`.
      5. Assert non-empty string response.

    Deferred body: implementation is CI-dependent. The scaffold is
    checked in so `pytest -m slow` has a concrete test to collect.
    """
    try:
        from tests.fixtures.tiny_model import tiny_model_path

        tiny_model_path()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tiny-model fixture unavailable: {exc}")

    pytest.xfail("train+prompt integration scaffolded; body deferred to first CI slow run")


@pytest.mark.slow
def test_qlora_crossplatform_dequantize() -> None:
    """Audit F05: a QLoRA-trained adapter loads on a non-CUDA host via dequantize.

    Shape:
      1. CI matrix has a CUDA job that trains QLoRA and uploads the
         adapter as an artifact.
      2. This test runs on a CPU-only matrix row; downloads the
         artifact, resolves `InferencePlan`, asserts
         `plan.dequantize_on_load is True`, loads, generates,
         asserts non-empty coherent output.

    Until the CUDA-producing job exists, the test is xfailed so it
    shows up in `pytest -m slow` as "expected failure pending CI".
    """
    pytest.xfail("needs CI artifact-sharing between CUDA and CPU jobs; F05 regression")
