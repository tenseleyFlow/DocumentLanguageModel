"""Inference-side gate — load, embed, weight computation.

Uses a minimal stub tokenizer + base model so tests don't need HF
downloads. Embedding math is real torch; only the upstream HF
surface is mocked.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dlm.inference.gate import (
    GateHandle,
    embed_prompt,
    load_gate_handle,
    uniform_weights,
    weights_for_prompt,
)
from dlm.train.gate import GateTrainingSample, train_gate


def _store(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(root=tmp_path)


class _StubTokenizer:
    """Returns a fixed token-id sequence for any prompt. Shape only
    matters for the embed pipeline — the base model's hidden states
    carry the actual routing signal."""

    def __init__(self, seq_len: int = 8) -> None:
        self._seq_len = seq_len

    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 512,
    ) -> dict[str, object]:
        import torch

        # Hash the prompt into a small id range so different prompts
        # produce different token sequences (otherwise the base model's
        # output is identical across inputs).
        seed = sum(ord(c) for c in prompt) % 1000
        torch.manual_seed(seed)
        ids = torch.randint(0, 100, (1, self._seq_len))
        mask = torch.ones(1, self._seq_len, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}


class _StubBaseModel:
    """Emits a last-hidden-state of a configurable shape. The hidden
    state is a linear projection of the input-ids so routing signals
    stay input-conditioned."""

    def __init__(self, *, hidden_dim: int = 16, vocab: int = 100) -> None:
        import torch

        self.hidden_dim = hidden_dim
        torch.manual_seed(0)
        self._embed = torch.randn(vocab, hidden_dim)

    def parameters(self):  # type: ignore[no-untyped-def]
        return iter([self._embed])

    def __call__(
        self,
        *,
        input_ids: object,
        attention_mask: object,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> object:
        import torch

        assert isinstance(input_ids, torch.Tensor)
        # Lookup embeddings: (batch, seq, hidden).
        flat = input_ids.clamp(max=self._embed.shape[0] - 1)
        hidden = self._embed[flat]
        return SimpleNamespace(hidden_states=(hidden,))

    forward = __call__


class _NoMaskTokenizer(_StubTokenizer):
    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 512,
    ) -> dict[str, object]:
        import torch

        ids = torch.randint(0, 100, (1, self._seq_len))
        return {"input_ids": ids}


class _NoParamBaseModel(_StubBaseModel):
    def parameters(self):  # type: ignore[no-untyped-def]
        return iter(())


def _train_gate_on_store(
    tmp_path: Path,
    *,
    adapter_names: tuple[str, ...] = ("a", "b"),
    input_dim: int = 16,
) -> SimpleNamespace:
    """Train a tiny gate with well-separated clusters so routing is
    deterministic for the test."""
    import torch

    store = _store(tmp_path)
    torch.manual_seed(0)
    center_a = torch.ones(input_dim)
    center_b = -torch.ones(input_dim)
    samples: list[GateTrainingSample] = []
    for _ in range(16):
        samples.append(
            GateTrainingSample(
                embedding=center_a + 0.05 * torch.randn(input_dim),
                adapter_name=adapter_names[0],
            )
        )
        samples.append(
            GateTrainingSample(
                embedding=center_b + 0.05 * torch.randn(input_dim),
                adapter_name=adapter_names[1],
            )
        )
    train_gate(
        store,  # type: ignore[arg-type]
        samples,
        adapter_names=adapter_names,
        input_dim=input_dim,
        hidden_proj_dim=16,
        steps=200,
        lr=3e-3,
        cold_start_floor=1,
        batch_size=16,
    )
    return store


class TestUniformHelper:
    def test_split_across_adapters(self) -> None:
        w = uniform_weights(("a", "b", "c", "d"))
        assert w == {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}

    def test_empty_tuple(self) -> None:
        assert uniform_weights(()) == {}


class TestEmbedPrompt:
    def test_returns_hidden_dim_vector(self) -> None:
        import torch

        tokenizer = _StubTokenizer()
        model = _StubBaseModel(hidden_dim=16)
        embedding = embed_prompt(prompt="hello", tokenizer=tokenizer, base_model=model)
        assert embedding.shape == (16,)
        assert embedding.dtype == torch.float32

    def test_different_prompts_yield_different_embeddings(self) -> None:
        import torch

        tokenizer = _StubTokenizer()
        model = _StubBaseModel(hidden_dim=16)
        e1 = embed_prompt(prompt="compute dgemm", tokenizer=tokenizer, base_model=model)
        e2 = embed_prompt(prompt="hello world", tokenizer=tokenizer, base_model=model)
        assert not torch.allclose(e1, e2)

    def test_falls_back_to_cpu_when_model_has_no_parameters(self) -> None:
        embedding = embed_prompt(
            prompt="hello",
            tokenizer=_StubTokenizer(),
            base_model=_NoParamBaseModel(hidden_dim=8),
        )
        assert embedding.shape == (8,)

    def test_mean_pools_without_attention_mask(self) -> None:
        embedding = embed_prompt(
            prompt="hello",
            tokenizer=_NoMaskTokenizer(),
            base_model=_StubBaseModel(hidden_dim=8),
        )
        assert embedding.shape == (8,)


class TestLoadGateHandle:
    def test_uniform_handle_from_cold_start(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        train_gate(
            store,  # type: ignore[arg-type]
            [GateTrainingSample(embedding=_tensor([0.1]), adapter_name="a")],
            adapter_names=["a", "b"],
            input_dim=1,
            cold_start_floor=4,
        )
        handle = load_gate_handle(store)  # type: ignore[arg-type]
        assert handle.is_uniform is True
        assert handle.gate is None
        assert handle.adapter_names == ("a", "b")

    def test_trained_handle(self, tmp_path: Path) -> None:
        store = _train_gate_on_store(tmp_path)
        handle = load_gate_handle(store)  # type: ignore[arg-type]
        assert handle.is_uniform is False
        assert handle.gate is not None
        assert handle.adapter_names == ("a", "b")


class TestWeightsForPrompt:
    def test_uniform_short_circuits(self, tmp_path: Path) -> None:
        """Uniform handle returns 1/N without touching tokenizer or model.
        We verify by passing ``None`` — a Real call would AttributeError."""
        handle = GateHandle(
            gate=None,
            metadata=_make_uniform_meta(("a", "b", "c")),
        )
        weights = weights_for_prompt(
            handle,
            prompt="anything",
            tokenizer=None,  # type: ignore[arg-type]
            base_model=None,  # type: ignore[arg-type]
        )
        assert weights == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})

    def test_trained_gate_sums_to_one(self, tmp_path: Path) -> None:
        store = _train_gate_on_store(tmp_path)
        handle = load_gate_handle(store)  # type: ignore[arg-type]
        tokenizer = _StubTokenizer()
        model = _StubBaseModel(hidden_dim=16)
        weights = weights_for_prompt(handle, prompt="any", tokenizer=tokenizer, base_model=model)
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_embedding_dim_mismatch_refused(self, tmp_path: Path) -> None:
        store = _train_gate_on_store(tmp_path, input_dim=16)
        handle = load_gate_handle(store)  # type: ignore[arg-type]
        # Model with a different hidden_dim than the gate expects.
        model = _StubBaseModel(hidden_dim=8)
        with pytest.raises(
            __import__("dlm.train.gate.errors", fromlist=["GateConfigError"]).GateConfigError,
            match="input_dim",
        ):
            weights_for_prompt(
                handle,
                prompt="any",
                tokenizer=_StubTokenizer(),
                base_model=model,
            )


def _tensor(values: list[float]) -> object:
    import torch

    return torch.tensor(values, dtype=torch.float32)


def _make_uniform_meta(adapter_names: tuple[str, ...]) -> object:
    from dlm.train.gate.module import GateMetadata

    return GateMetadata(
        input_dim=1,
        hidden_proj_dim=1,
        adapter_names=adapter_names,
        mode="uniform",
    )
