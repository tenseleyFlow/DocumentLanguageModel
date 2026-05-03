"""Key mapping for PEFT → MLX LoRA adapter conversion."""

from __future__ import annotations

from typing import Any

import pytest

from dlm.inference.mlx_adapter import (
    MlxConversionError,
    assert_mlx_adapter_applied,
    map_all_keys,
    map_peft_key_to_mlx,
)


class TestMapPeftKey:
    def test_lora_a_lowercases_and_strips_weight(self) -> None:
        got = map_peft_key_to_mlx("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight")
        assert got == "model.layers.0.self_attn.q_proj.lora_a"

    def test_lora_b_lowercases_and_strips_weight(self) -> None:
        got = map_peft_key_to_mlx("base_model.model.model.layers.5.mlp.down_proj.lora_B.weight")
        assert got == "model.layers.5.mlp.down_proj.lora_b"

    def test_base_model_prefix_stripped_once_only(self) -> None:
        # Inner `model.` (HF attribute) survives; only the outer
        # `base_model.model.` wrapper is dropped.
        got = map_peft_key_to_mlx("base_model.model.model.embed_tokens.lora_A.weight")
        assert got == "model.embed_tokens.lora_a"

    def test_non_lora_key_returns_none(self) -> None:
        # modules_to_save duplicates, bias tensors, etc.
        assert (
            map_peft_key_to_mlx(
                "base_model.model.model.embed_tokens.modules_to_save.default.weight"
            )
            is None
        )
        assert map_peft_key_to_mlx("something.else.bias") is None

    def test_bare_bias_key_returns_none(self) -> None:
        assert map_peft_key_to_mlx("base_model.model.model.layers.0.self_attn.q_proj.bias") is None


class TestMapAllKeys:
    def test_pair_mapping(self) -> None:
        keys = [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.model.layers.1.mlp.up_proj.lora_A.weight",
            "base_model.model.model.layers.1.mlp.up_proj.lora_B.weight",
        ]
        mapping = map_all_keys(keys)
        assert len(mapping) == 4
        assert mapping[keys[0]] == "model.layers.0.self_attn.q_proj.lora_a"
        assert mapping[keys[1]] == "model.layers.0.self_attn.q_proj.lora_b"

    def test_non_lora_keys_skipped_silently(self) -> None:
        keys = [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.model.embed_tokens.modules_to_save.default.weight",
        ]
        mapping = map_all_keys(keys)
        assert len(mapping) == 2

    def test_empty_adapter_raises(self) -> None:
        with pytest.raises(MlxConversionError, match="no LoRA A/B"):
            map_all_keys(["just.a.bias"])

    def test_duplicate_mlx_key_raises(self) -> None:
        # Two keys that both resolve to `q_proj.lora_a` after the
        # outer `base_model.model.` strip — one already unwrapped,
        # one wrapped. Defensive branch that matters if PEFT ever
        # emits the same tensor under both wrapped + unwrapped names.
        collision = [
            "q_proj.lora_A.weight",
            "base_model.model.q_proj.lora_A.weight",
        ]
        with pytest.raises(MlxConversionError, match="map to the same"):
            map_all_keys(collision)


class TestBuildMlxAdapterConfig:
    def test_non_positive_layer_count_rejected(self) -> None:
        from dlm.inference.mlx_adapter import build_mlx_adapter_config

        with pytest.raises(MlxConversionError, match="expected >=1"):
            build_mlx_adapter_config(
                {
                    "r": 8,
                    "target_modules": ["q_proj"],
                },
                0,
            )

    def test_attn_target_modules_get_self_attn_prefix(self) -> None:
        """mlx-lm matches `named_modules()` keys *inside* a transformer
        block via exact equality. PEFT's bare `q_proj` doesn't match
        the `self_attn.q_proj` FQN, so without the rewrite mlx-lm
        silently leaves the model un-wrapped — the textbook "trained
        model behaves like base" failure mode."""
        from dlm.inference.mlx_adapter import build_mlx_adapter_config

        cfg = build_mlx_adapter_config(
            {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            base_num_hidden_layers=28,
        )
        assert cfg["lora_parameters"]["keys"] == [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]

    def test_mlp_target_modules_get_mlp_prefix(self) -> None:
        from dlm.inference.mlx_adapter import build_mlx_adapter_config

        cfg = build_mlx_adapter_config(
            {
                "r": 8,
                "target_modules": ["gate_proj", "up_proj", "down_proj"],
            },
            base_num_hidden_layers=12,
        )
        assert cfg["lora_parameters"]["keys"] == [
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]

    def test_already_qualified_keys_pass_through(self) -> None:
        """Callers that pre-qualify (e.g. for non-decoder architectures)
        should not see their dotted keys re-rewritten."""
        from dlm.inference.mlx_adapter import build_mlx_adapter_config

        cfg = build_mlx_adapter_config(
            {
                "r": 8,
                "target_modules": ["self_attn.q_proj", "encoder.fc1"],
            },
            base_num_hidden_layers=12,
        )
        assert cfg["lora_parameters"]["keys"] == ["self_attn.q_proj", "encoder.fc1"]

    def test_unknown_target_module_passes_through_unqualified(self) -> None:
        """Names that aren't in the attn/mlp tables stay bare. Caller
        supervision is the user's responsibility — we don't guess."""
        from dlm.inference.mlx_adapter import build_mlx_adapter_config

        cfg = build_mlx_adapter_config(
            {
                "r": 8,
                "target_modules": ["unknown_proj"],
            },
            base_num_hidden_layers=12,
        )
        assert cfg["lora_parameters"]["keys"] == ["unknown_proj"]


class TestPeftSafetensorsToMlxTransposes:
    """PEFT and MLX-LM use different storage layouts for LoRA tensors:

      PEFT lora_A : [r, in_features]       MLX lora_a : [in_features, r]
      PEFT lora_B : [out_features, r]      MLX lora_b : [r, out_features]

    Without transposing, mlx-lm's `model.load_weights(strict=False)`
    silently skips the mismatched shapes and the adapter has no effect.
    """

    def test_lora_a_and_b_get_transposed(self, tmp_path: object) -> None:
        from pathlib import Path as _Path

        import torch
        from safetensors.torch import load_file, save_file

        from dlm.inference.mlx_adapter import peft_safetensors_to_mlx_safetensors

        tmp_path = _Path(str(tmp_path))
        peft_dir = tmp_path / "peft"
        peft_dir.mkdir()
        # PEFT shapes: lora_A=[r=4, in=8], lora_B=[out=16, r=4]
        peft_tensors = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.arange(
                32, dtype=torch.float32
            ).reshape(4, 8),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.arange(
                64, dtype=torch.float32
            ).reshape(16, 4),
        }
        save_file(peft_tensors, str(peft_dir / "adapter_model.safetensors"))

        mlx_path = tmp_path / "out" / "adapters.safetensors"
        peft_safetensors_to_mlx_safetensors(peft_dir, mlx_path)

        mlx_tensors = load_file(str(mlx_path))
        a = mlx_tensors["model.layers.0.self_attn.q_proj.lora_a"]
        b = mlx_tensors["model.layers.0.self_attn.q_proj.lora_b"]
        # Transposed shapes
        assert tuple(a.shape) == (8, 4)
        assert tuple(b.shape) == (4, 16)
        # Values match a transpose, not just a reshape.
        assert torch.equal(a, peft_tensors[next(iter(peft_tensors))].t())


class TestAssertMlxAdapterApplied:
    """Fail-loud post-load guard. mlx-lm silently leaves a model
    un-wrapped when keys don't match; this check turns that footgun
    into an explicit `MlxConversionError` so users see the failure
    rather than getting silent base-model output."""

    def _fake_model_with_params(self, names: list[str]) -> Any:
        """Build a stand-in for an mlx model that exposes
        `trainable_parameters()` returning a flat dict of fake tensors.
        We don't go through `mlx.utils.tree_flatten`'s real
        implementation here — assert_mlx_adapter_applied uses it
        directly, so we assert via the import-mock approach below."""

        class _FakeArr:
            shape = (1,)

        class _FakeModel:
            def trainable_parameters(self) -> dict[str, Any]:
                return {n: _FakeArr() for n in names}

        return _FakeModel()

    def test_passes_when_lora_params_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Stub mlx.utils.tree_flatten so the test doesn't require
        # mlx-lm's real flatten semantics — we only need it to walk
        # the dict-shaped trainable_parameters() output.
        import sys
        import types as _types

        fake_mlx = _types.ModuleType("mlx")
        fake_mlx_utils = _types.ModuleType("mlx.utils")

        def _tree_flatten(d: dict[str, Any]) -> list[tuple[str, Any]]:
            return list(d.items())

        fake_mlx_utils.tree_flatten = _tree_flatten  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
        monkeypatch.setitem(sys.modules, "mlx.utils", fake_mlx_utils)

        model = self._fake_model_with_params(
            [
                "model.layers.0.self_attn.q_proj.lora_a",
                "model.layers.0.self_attn.q_proj.lora_b",
            ]
        )
        # Should not raise.
        assert_mlx_adapter_applied(model, expected_keys=["self_attn.q_proj"])

    def test_raises_when_no_lora_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types as _types

        fake_mlx = _types.ModuleType("mlx")
        fake_mlx_utils = _types.ModuleType("mlx.utils")
        fake_mlx_utils.tree_flatten = lambda d: list(d.items())  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
        monkeypatch.setitem(sys.modules, "mlx.utils", fake_mlx_utils)

        # Only base parameters; no lora_a/lora_b.
        model = self._fake_model_with_params(
            [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ]
        )
        with pytest.raises(MlxConversionError, match="zero `lora_a`"):
            assert_mlx_adapter_applied(model, expected_keys=["self_attn.q_proj"])
