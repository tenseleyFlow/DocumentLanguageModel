"""`build_lora_config` — target_modules + modules_to_save wiring."""

from __future__ import annotations

from dlm.base_models import BASE_MODELS
from dlm.train.adapter import build_lora_config


class TestBuildLoraConfig:
    def test_split_qkv_target_modules_pass_through(self) -> None:
        spec = BASE_MODELS["smollm2-135m"]
        cfg = build_lora_config(
            spec, lora_r=16, lora_alpha=32, lora_dropout=0.05, tokenizer_grew=False
        )
        assert set(cfg.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.05
        assert cfg.modules_to_save is None

    def test_fused_qkv_for_phi3(self) -> None:
        spec = BASE_MODELS["phi-3.5-mini"]
        cfg = build_lora_config(
            spec, lora_r=8, lora_alpha=16, lora_dropout=0.1, tokenizer_grew=False
        )
        assert "qkv_proj" in cfg.target_modules

    def test_tokenizer_grew_sets_modules_to_save(self) -> None:
        spec = BASE_MODELS["qwen2.5-0.5b"]
        cfg = build_lora_config(
            spec, lora_r=8, lora_alpha=16, lora_dropout=0.0, tokenizer_grew=True
        )
        assert cfg.modules_to_save == ["embed_tokens", "lm_head"]

    def test_bias_and_task_type(self) -> None:
        """Bias `none` + `CAUSAL_LM` task are the LoRA defaults we want."""
        from peft import TaskType

        spec = BASE_MODELS["smollm2-135m"]
        cfg = build_lora_config(
            spec, lora_r=8, lora_alpha=16, lora_dropout=0.0, tokenizer_grew=False
        )
        assert cfg.bias == "none"
        assert cfg.task_type == TaskType.CAUSAL_LM
