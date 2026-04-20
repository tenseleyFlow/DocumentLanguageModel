"""ROCm training smoke (Sprint 22).

Verifies the doctor→plan→trainer pipeline actually runs on a ROCm
host without the refusal matrix blocking LoRA. Uses the tiny-model
session fixture and runs for a single step; the smoke is that
`run_training` returns a result rather than raising.

Skipped unless:
- `torch.version.hip` is truthy at runtime (real ROCm torch build)
- `DLM_ENABLE_ROCM_SMOKE=1` in the environment (opt-in even on a
  ROCm host so local `pytest -m slow` runs stay CPU/CUDA-only)

CI: no default runner exists; expected to be run on a self-hosted
ROCm box via a scheduled workflow. Documented in
`docs/hardware/rocm.md`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.fixtures.trained_store import TrainedStoreHandle


def _rocm_host() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    return bool(getattr(torch.version, "hip", None))


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _rocm_host(), reason="requires a ROCm PyTorch build"),
    pytest.mark.skipif(
        os.environ.get("DLM_ENABLE_ROCM_SMOKE") != "1",
        reason="set DLM_ENABLE_ROCM_SMOKE=1 to opt in to the ROCm smoke on a real host",
    ),
]


def test_rocm_lora_smoke_runs(  # pragma: no cover - gpu+rocm path
    trained_store: TrainedStoreHandle,
) -> None:
    """One-step LoRA train on ROCm — no refusal, produces an adapter version bump."""
    # The `trained_store` session fixture trained once during setup;
    # reaching this point on a ROCm host without a refusal is the
    # smoke signal. Assert the store has at least one committed
    # adapter version AND that the lock recorded the ROCm tier —
    # audit-08 N8 catches a smoke that passes on a CPU pytest run
    # that never touched ROCm.
    store = trained_store.store
    adapter_dir = store.resolve_current_adapter()
    assert adapter_dir is not None, (
        "trained_store fixture produced no adapter on ROCm — "
        "LoRA path likely blocked by a refusal that shouldn't fire"
    )
    assert (adapter_dir / "adapter_model.safetensors").exists(), (
        "ROCm LoRA wrote the pointer but not the adapter weights"
    )

    # Hardware-tier contract: the lock must record ROCm for this smoke
    # to actually prove the ROCm path was exercised.
    from dlm.lock import load_lock

    lock = load_lock(store.root)
    assert lock is not None, "trained_store did not persist a dlm.lock"
    assert lock.hardware_tier == "rocm", (
        f"trained_store produced hardware_tier={lock.hardware_tier!r}; "
        "expected 'rocm'. This smoke ran on the wrong host."
    )
