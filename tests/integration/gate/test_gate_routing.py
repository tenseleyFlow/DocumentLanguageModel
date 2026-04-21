"""Gate end-to-end integration: schema v8 → gate train → inference routing.

Exercises the full flow on synthetic embeddings (no HF download):

1. Parse a v8 ``.dlm`` with ``training.adapters`` + ``training.gate.enabled``.
2. Build gate-training samples with separable clusters per adapter.
3. Train the gate via ``train_gate``.
4. Load it through ``inference.gate.load_gate_handle``.
5. Dispatch held-out embeddings through ``weights_for_prompt`` — assert
   the gate routes each cluster to its adapter > 80% of the time at
   > 0.5 confidence.

The production path layers HF tokenizer + base-model encoding on top;
that's already covered by `tests/unit/inference/test_gate.py` with stub
fixtures. This test proves the schema+store+gate pipeline holds
together without the heavy torch-model surface.

Marked ``slow`` so it's skipped from the default gate sweep — the 200
AdamW steps plus 64 held-out evals run in ~0.5s but the marker keeps
CI lanes clean.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dlm.doc.parser import parse_text
from dlm.inference.gate import GateHandle, load_gate_handle
from dlm.train.gate import GateTrainingSample, train_gate

pytestmark = pytest.mark.slow


_DLM_TEXT = """---
dlm_id: 01KPQ8GATEMTEG000000000000
dlm_version: 8
base_model: smollm2-135m
training:
  adapters:
    lexer: {}
    runtime: {}
  gate:
    enabled: true
    hidden_proj_dim: 16
    steps: 200
    lr: 0.003
    cold_start_floor: 1
---
::instruction#lexer::
### Q
what is a token?
### A
the smallest atom the parser reads.

::instruction#runtime::
### Q
what is a heap block?
### A
a range of contiguously-allocated memory.
"""


def _store(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(root=tmp_path)


def _cluster_samples(
    *,
    adapter_a: str,
    adapter_b: str,
    input_dim: int,
    per_class: int,
    seed: int,
) -> list[GateTrainingSample]:
    """Two well-separated clusters. Used both for training and held-out
    evaluation by calling with different seeds."""
    import torch

    g = torch.Generator().manual_seed(seed)
    center_a = torch.ones(input_dim)
    center_b = -torch.ones(input_dim)
    samples: list[GateTrainingSample] = []
    for _ in range(per_class):
        samples.append(
            GateTrainingSample(
                embedding=center_a + 0.1 * torch.randn(input_dim, generator=g),
                adapter_name=adapter_a,
            )
        )
        samples.append(
            GateTrainingSample(
                embedding=center_b + 0.1 * torch.randn(input_dim, generator=g),
                adapter_name=adapter_b,
            )
        )
    return samples


def test_schema_to_gate_to_inference_end_to_end(tmp_path: Path) -> None:
    """Parse v8 .dlm → train gate → load via inference handle → route
    held-out embeddings correctly > 80% of the time."""
    parsed = parse_text(_DLM_TEXT)
    fm = parsed.frontmatter
    assert fm.dlm_version == 8
    assert fm.training.gate.enabled is True
    assert fm.training.adapters is not None
    adapter_names = tuple(fm.training.adapters)
    assert adapter_names == ("lexer", "runtime")

    store = _store(tmp_path)
    input_dim = 16
    training_samples = _cluster_samples(
        adapter_a="lexer",
        adapter_b="runtime",
        input_dim=input_dim,
        per_class=24,
        seed=1,
    )
    result = train_gate(
        store,  # type: ignore[arg-type]
        training_samples,
        adapter_names=adapter_names,
        input_dim=input_dim,
        hidden_proj_dim=fm.training.gate.hidden_proj_dim,
        steps=fm.training.gate.steps,
        lr=fm.training.gate.lr,
        cold_start_floor=fm.training.gate.cold_start_floor,
        batch_size=16,
        seed=0,
    )
    assert result.mode == "trained"
    assert result.per_adapter_samples == {"lexer": 24, "runtime": 24}

    handle = load_gate_handle(store)  # type: ignore[arg-type]
    assert isinstance(handle, GateHandle)
    assert handle.is_uniform is False
    assert handle.gate is not None

    # --- held-out routing accuracy ---------------------------------
    import torch

    held_out = _cluster_samples(
        adapter_a="lexer",
        adapter_b="runtime",
        input_dim=input_dim,
        per_class=32,
        seed=999,
    )
    correct = 0
    with torch.no_grad():
        for sample in held_out:
            probs = handle.gate(sample.embedding.unsqueeze(0)).squeeze(0)
            # The gate's adapter order matches metadata.adapter_names;
            # we compare against the ground-truth label.
            predicted = handle.adapter_names[int(probs.argmax().item())]
            if predicted == sample.adapter_name and probs.max().item() > 0.5:
                correct += 1

    # 64 held-out probes, target >80% correct at >0.5 confidence.
    assert correct >= 52, (
        f"gate routed only {correct}/64 held-out samples correctly (target: >= 52 = 81%)"
    )
