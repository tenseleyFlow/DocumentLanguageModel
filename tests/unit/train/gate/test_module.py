"""Gate nn.Module — shape + parameter count + metadata round-trip."""

from __future__ import annotations

import pytest

from dlm.train.gate import Gate, GateConfigError, GateMetadata


class TestGateConstruction:
    def test_forward_shape(self) -> None:
        import torch

        gate = Gate(input_dim=128, hidden_proj_dim=32, n_adapters=3)
        x = torch.randn(5, 128)
        y = gate(x)
        assert y.shape == (5, 3)
        # Softmax rows sum to ~1.
        assert torch.allclose(y.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_parameter_count(self) -> None:
        gate = Gate(input_dim=2048, hidden_proj_dim=64, n_adapters=4)
        expected = (2048 * 64 + 64) + (64 * 4 + 4)
        assert gate.num_parameters() == expected

    def test_batch_dim_preserved(self) -> None:
        import torch

        gate = Gate(input_dim=16, hidden_proj_dim=8, n_adapters=2)
        # 3D input (batch, time, features).
        x = torch.randn(2, 7, 16)
        y = gate(x)
        assert y.shape == (2, 7, 2)

    def test_single_adapter_refused(self) -> None:
        with pytest.raises(ValueError, match="n_adapters must be >= 2"):
            Gate(input_dim=8, hidden_proj_dim=4, n_adapters=1)

    def test_nonpositive_dims_refused(self) -> None:
        with pytest.raises(ValueError, match="input_dim"):
            Gate(input_dim=0, hidden_proj_dim=4, n_adapters=2)
        with pytest.raises(ValueError, match="hidden_proj_dim"):
            Gate(input_dim=8, hidden_proj_dim=0, n_adapters=2)


class TestGateMetadataJson:
    def test_round_trip(self) -> None:
        meta = GateMetadata(
            input_dim=512,
            hidden_proj_dim=64,
            adapter_names=("lexer", "runtime"),
            mode="trained",
            entropy_lambda=0.02,
        )
        raw = meta.to_json()
        restored = GateMetadata.from_json(raw)
        assert restored == meta

    def test_missing_required_field(self) -> None:
        with pytest.raises(GateConfigError, match="missing fields"):
            GateMetadata.from_json({"input_dim": 8, "hidden_proj_dim": 4, "mode": "trained"})

    def test_bad_mode(self) -> None:
        with pytest.raises(GateConfigError, match="mode"):
            GateMetadata.from_json(
                {
                    "input_dim": 8,
                    "hidden_proj_dim": 4,
                    "adapter_names": ["a", "b"],
                    "mode": "bogus",
                }
            )

    def test_adapter_names_not_list(self) -> None:
        with pytest.raises(GateConfigError, match="adapter_names"):
            GateMetadata.from_json(
                {
                    "input_dim": 8,
                    "hidden_proj_dim": 4,
                    "adapter_names": "not-a-list",
                    "mode": "trained",
                }
            )
