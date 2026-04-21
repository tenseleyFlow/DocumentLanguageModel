"""Gate trainer — cold-start fallback, convergence, persistence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dlm.train.gate import (
    GateConfigError,
    GateTrainingSample,
    load_gate,
    train_gate,
)
from dlm.train.gate.paths import gate_config_path, gate_save_path


def _store(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(root=tmp_path)


def _synthetic_samples(
    *,
    per_class: int,
    input_dim: int,
    seed: int = 0,
) -> list[GateTrainingSample]:
    """Two well-separated clusters: class 'a' centered at +1, 'b' at -1.
    A tiny MLP converges on this in well under 200 steps."""
    import torch

    g = torch.Generator().manual_seed(seed)
    samples: list[GateTrainingSample] = []
    center_a = torch.ones(input_dim)
    center_b = -torch.ones(input_dim)
    for _ in range(per_class):
        samples.append(
            GateTrainingSample(
                embedding=center_a + 0.1 * torch.randn(input_dim, generator=g),
                adapter_name="a",
            )
        )
        samples.append(
            GateTrainingSample(
                embedding=center_b + 0.1 * torch.randn(input_dim, generator=g),
                adapter_name="b",
            )
        )
    return samples


class TestColdStartFallback:
    def test_below_floor_writes_uniform_config(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        samples = [
            GateTrainingSample(embedding=_tensor([0.1]), adapter_name="a"),
            GateTrainingSample(embedding=_tensor([0.2]), adapter_name="a"),
            # adapter 'b' has zero samples → below floor
        ]
        result = train_gate(
            store,  # type: ignore[arg-type]
            samples,
            adapter_names=["a", "b"],
            input_dim=1,
            cold_start_floor=4,
            steps=200,
        )
        assert result.mode == "uniform"
        assert result.per_adapter_samples == {"a": 2, "b": 0}
        assert gate_config_path(store).exists()  # type: ignore[arg-type]
        # Weights file should NOT exist for uniform mode.
        assert not gate_save_path(store).exists()  # type: ignore[arg-type]
        raw = json.loads(gate_config_path(store).read_text(encoding="utf-8"))  # type: ignore[arg-type]
        assert raw["mode"] == "uniform"
        assert raw["adapter_names"] == ["a", "b"]

    def test_single_adapter_refused(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(GateConfigError, match=">= 2 adapters"):
            train_gate(
                store,  # type: ignore[arg-type]
                [],
                adapter_names=["only"],
                input_dim=4,
            )

    def test_duplicate_adapter_names_refused(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(GateConfigError, match="unique adapter names"):
            train_gate(
                store,  # type: ignore[arg-type]
                [],
                adapter_names=["a", "b", "a"],
                input_dim=4,
            )

    def test_wrong_embedding_dim_refused(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        samples = _synthetic_samples(per_class=5, input_dim=4)
        with pytest.raises(GateConfigError, match="embedding dim"):
            train_gate(
                store,  # type: ignore[arg-type]
                samples,
                adapter_names=["a", "b"],
                input_dim=16,  # mismatch
                cold_start_floor=1,
                steps=5,
            )


class TestConvergence:
    def test_two_cluster_task_converges(self, tmp_path: Path) -> None:
        import torch

        store = _store(tmp_path)
        samples = _synthetic_samples(per_class=32, input_dim=16, seed=1)
        result = train_gate(
            store,  # type: ignore[arg-type]
            samples,
            adapter_names=["a", "b"],
            input_dim=16,
            hidden_proj_dim=16,
            steps=200,
            lr=3e-3,
            cold_start_floor=1,
            batch_size=16,
            seed=0,
        )
        assert result.mode == "trained"
        assert result.final_loss is not None
        # Held-out predictions: the gate should assign > 0.8 mass to
        # the correct class on unseen samples from each cluster.
        gate, meta = load_gate(store)  # type: ignore[arg-type]
        assert gate is not None
        assert meta.mode == "trained"
        held_out = _synthetic_samples(per_class=16, input_dim=16, seed=99)
        correct = 0
        with torch.no_grad():
            for s in held_out:
                idx = meta.adapter_names.index(s.adapter_name)
                probs = gate(s.embedding.unsqueeze(0))
                if probs[0, idx].item() > 0.8:
                    correct += 1
        # 2 clusters × 16 held-out = 32 probes. Allow a little slack.
        assert correct >= 28, f"only {correct}/32 held-out predictions passed 0.8 threshold"

    def test_observer_hook_fires(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        samples = _synthetic_samples(per_class=5, input_dim=4, seed=0)
        seen: list[tuple[int, float, float]] = []
        train_gate(
            store,  # type: ignore[arg-type]
            samples,
            adapter_names=["a", "b"],
            input_dim=4,
            steps=7,
            cold_start_floor=1,
            batch_size=4,
            on_step=lambda step, loss, entropy: seen.append((step, loss, entropy)),
        )
        assert len(seen) == 7


class TestLoadGateErrors:
    def test_missing_config(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(GateConfigError, match="not found"):
            load_gate(store)  # type: ignore[arg-type]

    def test_trained_mode_missing_weights(self, tmp_path: Path) -> None:
        """Trained-mode config without a weights file is a broken store."""
        store = _store(tmp_path)
        gate_config_path(store).parent.mkdir(parents=True, exist_ok=True)  # type: ignore[arg-type]
        gate_config_path(store).write_text(  # type: ignore[arg-type]
            json.dumps(
                {
                    "input_dim": 4,
                    "hidden_proj_dim": 4,
                    "adapter_names": ["a", "b"],
                    "mode": "trained",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(GateConfigError, match="weights file"):
            load_gate(store)  # type: ignore[arg-type]

    def test_uniform_mode_returns_none_gate(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        samples = [GateTrainingSample(embedding=_tensor([0.1]), adapter_name="a")]
        train_gate(
            store,  # type: ignore[arg-type]
            samples,
            adapter_names=["a", "b"],
            input_dim=1,
            cold_start_floor=4,
        )
        gate, meta = load_gate(store)  # type: ignore[arg-type]
        assert gate is None
        assert meta.mode == "uniform"
        assert meta.adapter_names == ("a", "b")


class TestDeterminism:
    """Two runs with the same seed + samples produce bit-identical output.

    The gate trainer seeds `torch.manual_seed(seed)` for init,
    `torch.Generator().manual_seed(seed)` for batch sampling, and
    never touches non-deterministic ops. This test is the
    regression anchor: any future change that introduces
    non-determinism (dropout-without-seed, torch.rand without a
    generator, CUDA non-deterministic kernels) surfaces here.
    """

    def test_same_seed_same_weights_and_ewma_loss(self, tmp_path: Path) -> None:
        import torch
        from safetensors.torch import load_file

        samples = _synthetic_samples(per_class=32, input_dim=16, seed=1)

        store_a = _store(tmp_path / "a")
        store_b = _store(tmp_path / "b")
        kwargs: dict[str, object] = {
            "adapter_names": ["a", "b"],
            "input_dim": 16,
            "hidden_proj_dim": 16,
            "steps": 50,
            "lr": 3e-3,
            "cold_start_floor": 1,
            "batch_size": 16,
            "seed": 7,
        }
        result_a = train_gate(store_a, samples, **kwargs)  # type: ignore[arg-type]
        result_b = train_gate(store_b, samples, **kwargs)  # type: ignore[arg-type]

        assert result_a.mode == "trained"
        assert result_a.final_loss == result_b.final_loss
        assert result_a.final_entropy == result_b.final_entropy
        assert result_a.per_adapter_mean_weight == result_b.per_adapter_mean_weight

        weights_a = load_file(str(gate_save_path(store_a)))  # type: ignore[arg-type]
        weights_b = load_file(str(gate_save_path(store_b)))  # type: ignore[arg-type]
        assert set(weights_a.keys()) == set(weights_b.keys())
        for name, tensor_a in weights_a.items():
            assert torch.equal(tensor_a, weights_b[name]), (
                f"gate weight {name!r} differs between identical-seed runs"
            )

    def test_different_seed_different_weights(self, tmp_path: Path) -> None:
        """Sanity check: different seeds must not produce identical weights.

        Guards against a scenario where the seed parameter gets silently
        ignored (e.g., replaced by `torch.seed()`) and every run drifts
        off the same RNG state. If this test passes trivially with any
        seed, `test_same_seed_same_weights_and_ewma_loss` above would be
        vacuous.
        """
        import torch
        from safetensors.torch import load_file

        samples = _synthetic_samples(per_class=32, input_dim=16, seed=1)
        store_a = _store(tmp_path / "seed7")
        store_b = _store(tmp_path / "seed42")
        kwargs: dict[str, object] = {
            "adapter_names": ["a", "b"],
            "input_dim": 16,
            "hidden_proj_dim": 16,
            "steps": 50,
            "lr": 3e-3,
            "cold_start_floor": 1,
            "batch_size": 16,
        }
        train_gate(store_a, samples, seed=7, **kwargs)  # type: ignore[arg-type]
        train_gate(store_b, samples, seed=42, **kwargs)  # type: ignore[arg-type]
        weights_a = load_file(str(gate_save_path(store_a)))  # type: ignore[arg-type]
        weights_b = load_file(str(gate_save_path(store_b)))  # type: ignore[arg-type]
        any_differs = any(not torch.equal(weights_a[name], weights_b[name]) for name in weights_a)
        assert any_differs, "different seeds produced identical weights — seed is being ignored"


def _tensor(values: list[float]) -> object:
    import torch

    return torch.tensor(values, dtype=torch.float32)
