"""Sprint 40 smoke proof that the Mixtral row still flows through Sprint 34 gate paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dlm.base_models import BASE_MODELS
from dlm.inference.gate import GateHandle, load_gate_handle
from dlm.modality import modality_for
from dlm.train.gate import GateTrainingSample, train_gate


def _store(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(root=tmp_path)


def test_mixtral_text_moe_row_still_uses_text_gate_pipeline(tmp_path: Path) -> None:
    import torch

    spec = BASE_MODELS["mixtral-8x7b-instruct"]
    dispatch = modality_for(spec)
    assert spec.modality == "text-moe"
    assert dispatch.accepts_images is False
    assert dispatch.accepts_audio is False

    store = _store(tmp_path)
    samples: list[GateTrainingSample] = []
    for _ in range(12):
        samples.append(
            GateTrainingSample(embedding=torch.ones(8) + 0.05 * torch.randn(8), adapter_name="a")
        )
        samples.append(
            GateTrainingSample(embedding=-torch.ones(8) + 0.05 * torch.randn(8), adapter_name="b")
        )

    result = train_gate(
        store,  # type: ignore[arg-type]
        samples,
        adapter_names=("a", "b"),
        input_dim=8,
        hidden_proj_dim=8,
        steps=80,
        lr=3e-3,
        cold_start_floor=1,
        batch_size=8,
        seed=0,
    )
    assert result.mode == "trained"

    handle = load_gate_handle(store)  # type: ignore[arg-type]
    assert isinstance(handle, GateHandle)
    assert handle.is_uniform is False
    assert handle.metadata.adapter_names == ("a", "b")
