"""`apply_control` — forward_pre_hook attach/detach + arithmetic."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from dlm.control import ControlApplyError, apply_control


class _ToyLayer(nn.Module):
    """A stand-in for an HF decoder layer.

    Owns `self_attn.q_proj` so the dim-validation path has something
    to inspect; the forward just passes the hidden state through so
    the hook's perturbation is visible on the output.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        # `nn.Linear(hidden_dim, hidden_dim)` weight shape is
        # `(out, in)` — the `[-1]` the apply path reads is `in = hidden_dim`.
        self.self_attn.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _ToyModel(nn.Module):
    """Minimal HF-shaped wrapper: `model.model.layers[i]`."""

    def __init__(self, n_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_ToyLayer(hidden_dim) for _ in range(n_layers)])


def _run_through_layer(model: _ToyModel, layer_index: int, hidden: torch.Tensor) -> torch.Tensor:
    return model.model.layers[layer_index](hidden)


class TestHookArithmetic:
    def test_adds_scaled_vector_to_hidden(self) -> None:
        model = _ToyModel(n_layers=4, hidden_dim=8)
        vector = np.ones(8, dtype=np.float32)
        hidden = torch.zeros(1, 3, 8)
        with apply_control(model, vector, layer_index=2, strength=2.5):
            out = _run_through_layer(model, 2, hidden)
        # strength=2.5, vector=[1,1,1,...] → each output element is 2.5.
        assert torch.allclose(out, torch.full_like(out, 2.5))

    def test_zero_strength_is_passthrough(self) -> None:
        model = _ToyModel(n_layers=2, hidden_dim=4)
        vector = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        hidden = torch.randn(2, 5, 4)
        with apply_control(model, vector, layer_index=0, strength=0.0):
            out = _run_through_layer(model, 0, hidden)
        assert torch.allclose(out, hidden)

    def test_negative_strength_pushes_opposite(self) -> None:
        model = _ToyModel(n_layers=2, hidden_dim=3)
        vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        hidden = torch.zeros(1, 1, 3)
        with apply_control(model, vector, layer_index=0, strength=-1.5):
            out = _run_through_layer(model, 0, hidden)
        expected = torch.tensor([[[-1.5, 0.0, 0.0]]])
        assert torch.allclose(out, expected)

    def test_vector_broadcasts_across_batch_and_seq(self) -> None:
        model = _ToyModel(n_layers=1, hidden_dim=4)
        vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        hidden = torch.zeros(3, 7, 4)  # batch=3, seq=7
        with apply_control(model, vector, layer_index=0, strength=1.0):
            out = _run_through_layer(model, 0, hidden)
        # Every (batch, seq, 0) is 1.0; every (batch, seq, 1..3) is 0.
        assert torch.allclose(out[..., 0], torch.ones(3, 7))
        assert torch.allclose(out[..., 1:], torch.zeros(3, 7, 3))


class TestHookLifecycle:
    def test_hook_removed_on_clean_exit(self) -> None:
        model = _ToyModel(n_layers=1, hidden_dim=4)
        vector = np.ones(4, dtype=np.float32)
        hidden = torch.zeros(1, 1, 4)
        with apply_control(model, vector, layer_index=0, strength=1.0):
            pass
        # After the block, the layer should not perturb anymore.
        out = _run_through_layer(model, 0, hidden)
        assert torch.allclose(out, hidden)

    def test_hook_removed_on_exception(self) -> None:
        model = _ToyModel(n_layers=1, hidden_dim=4)
        vector = np.ones(4, dtype=np.float32)
        hidden = torch.zeros(1, 1, 4)
        with pytest.raises(RuntimeError, match="boom"):
            with apply_control(model, vector, layer_index=0, strength=1.0):
                raise RuntimeError("boom")
        # Hook must be gone even after an exception.
        out = _run_through_layer(model, 0, hidden)
        assert torch.allclose(out, hidden)


class TestValidation:
    def test_non_1d_vector_rejected(self) -> None:
        model = _ToyModel(n_layers=1, hidden_dim=4)
        vector = np.zeros((2, 4), dtype=np.float32)
        with pytest.raises(ControlApplyError, match="1D"):
            with apply_control(model, vector, layer_index=0):
                pass

    def test_non_finite_vector_rejected(self) -> None:
        model = _ToyModel(n_layers=1, hidden_dim=4)
        vector = np.array([1.0, float("nan"), 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ControlApplyError, match="non-finite"):
            with apply_control(model, vector, layer_index=0):
                pass

    def test_dim_mismatch_rejected(self) -> None:
        model = _ToyModel(n_layers=1, hidden_dim=4)
        vector = np.zeros(8, dtype=np.float32)  # wrong dim
        with pytest.raises(ControlApplyError, match="hidden dim"):
            with apply_control(model, vector, layer_index=0):
                pass

    def test_out_of_bounds_layer_rejected(self) -> None:
        model = _ToyModel(n_layers=2, hidden_dim=4)
        vector = np.ones(4, dtype=np.float32)
        with pytest.raises(ControlApplyError, match="out of bounds"):
            with apply_control(model, vector, layer_index=99):
                pass

    def test_negative_layer_index_works(self) -> None:
        # `-1` should select the last layer, matching list semantics.
        model = _ToyModel(n_layers=3, hidden_dim=4)
        vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        hidden = torch.zeros(1, 1, 4)
        with apply_control(model, vector, layer_index=-1, strength=2.0):
            out = model.model.layers[-1](hidden)
        assert out[0, 0, 0].item() == 2.0

    def test_model_without_layers_attribute_rejected(self) -> None:
        bare = nn.Linear(4, 4)
        vector = np.ones(4, dtype=np.float32)
        with pytest.raises(ControlApplyError, match="model.layers"):
            with apply_control(bare, vector, layer_index=0):
                pass
