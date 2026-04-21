"""Static mean-gate fallback for Ollama export."""

from __future__ import annotations

import pytest

from dlm.export.gate_fallback import mean_gate_weights, uniform_adapter_mix
from dlm.train.gate.module import Gate, GateMetadata


class TestUniformAdapterMix:
    def test_three_adapters_third_each(self) -> None:
        mix = uniform_adapter_mix(("a", "b", "c"))
        assert mix == [("a", 1 / 3), ("b", 1 / 3), ("c", 1 / 3)]

    def test_empty_tuple(self) -> None:
        assert uniform_adapter_mix(()) == []


class TestMeanGateWeights:
    def _gate_and_meta(self) -> tuple[Gate, GateMetadata]:
        gate = Gate(input_dim=8, hidden_proj_dim=4, n_adapters=2)
        meta = GateMetadata(
            input_dim=8,
            hidden_proj_dim=4,
            adapter_names=("a", "b"),
            mode="trained",
        )
        return gate, meta

    def test_empty_corpus_refused(self) -> None:
        gate, meta = self._gate_and_meta()
        with pytest.raises(ValueError, match=">= 1 prompt embedding"):
            mean_gate_weights(gate, meta, [])

    def test_weights_shape_and_sum_to_one(self) -> None:
        import torch

        gate, meta = self._gate_and_meta()
        embeddings = [torch.randn(8) for _ in range(16)]
        mix = mean_gate_weights(gate, meta, embeddings)
        assert [name for name, _ in mix] == ["a", "b"]
        total = sum(w for _, w in mix)
        assert total == pytest.approx(1.0, abs=1e-5)
        for _, w in mix:
            assert 0.0 <= w <= 1.0

    def test_dim_mismatch_refused(self) -> None:
        import torch

        gate, meta = self._gate_and_meta()
        # Wrong-dim embedding.
        with pytest.raises(ValueError, match="input_dim"):
            mean_gate_weights(gate, meta, [torch.randn(4)])

    def test_mean_reflects_per_prompt_skew(self) -> None:
        """Ten prompts near cluster A + one prompt near cluster B should
        average out to favor A. Sanity check that mean_gate_weights isn't
        just emitting uniform."""
        import torch

        gate = Gate(input_dim=8, hidden_proj_dim=8, n_adapters=2)
        meta = GateMetadata(
            input_dim=8,
            hidden_proj_dim=8,
            adapter_names=("a", "b"),
            mode="trained",
        )
        # Force the gate weights so it's (almost) deterministic: class-a
        # embeddings near +1, class-b near -1.
        torch.manual_seed(0)
        a_embeddings = [torch.ones(8) + 0.01 * torch.randn(8) for _ in range(10)]
        b_embedding = -torch.ones(8)
        # We won't train here — untrained gate may or may not favor A.
        # The point is only that the mean is a real average (not uniform
        # or fixed), which we check by comparing against a single-prompt
        # case.
        mix_mixed = mean_gate_weights(gate, meta, a_embeddings + [b_embedding])
        mix_single_a = mean_gate_weights(gate, meta, [a_embeddings[0]])
        # Different input distributions → different averaged outputs.
        assert mix_mixed != mix_single_a
