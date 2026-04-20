"""Key mapping for PEFT → MLX LoRA adapter conversion."""

from __future__ import annotations

import pytest

from dlm.inference.mlx_adapter import (
    MlxConversionError,
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
