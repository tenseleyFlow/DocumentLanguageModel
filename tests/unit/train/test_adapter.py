"""`build_lora_config` — target_modules + modules_to_save wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlm.base_models import BASE_MODELS
from dlm.train.adapter import build_lora_config, verify_resume_tokenizer_compat
from dlm.train.errors import ResumeIntegrityError


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


class TestVerifyResumeTokenizerCompat:
    """Audit-04 M5: cross-check saved adapter vs current tokenizer_grew."""

    def _write_adapter_config(self, path: Path, modules_to_save: list[str] | None) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text(
            json.dumps({"modules_to_save": modules_to_save})
        )

    def test_both_no_embeddings_passes(self, tmp_path: Path) -> None:
        self._write_adapter_config(tmp_path, modules_to_save=None)
        verify_resume_tokenizer_compat(tmp_path, tokenizer_grew=False)

    def test_both_with_embeddings_passes(self, tmp_path: Path) -> None:
        self._write_adapter_config(tmp_path, modules_to_save=["embed_tokens", "lm_head"])
        verify_resume_tokenizer_compat(tmp_path, tokenizer_grew=True)

    def test_grew_but_adapter_lacks_embeddings_raises(self, tmp_path: Path) -> None:
        """Current tokenizer grew, saved adapter didn't train embeddings."""
        self._write_adapter_config(tmp_path, modules_to_save=None)
        with pytest.raises(ResumeIntegrityError, match="vocab grew"):
            verify_resume_tokenizer_compat(tmp_path, tokenizer_grew=True)

    def test_adapter_has_embeddings_but_tokenizer_didnt_grow_raises(self, tmp_path: Path) -> None:
        """Adapter was trained with modules_to_save but current tokenizer doesn't need it."""
        self._write_adapter_config(tmp_path, modules_to_save=["embed_tokens", "lm_head"])
        with pytest.raises(ResumeIntegrityError, match="did not require"):
            verify_resume_tokenizer_compat(tmp_path, tokenizer_grew=False)

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ResumeIntegrityError, match="adapter_config.json"):
            verify_resume_tokenizer_compat(tmp_path, tokenizer_grew=False)

    def test_unreadable_config_raises(self, tmp_path: Path) -> None:
        (tmp_path / "adapter_config.json").write_text("not valid json {{{")
        with pytest.raises(ResumeIntegrityError, match="unreadable"):
            verify_resume_tokenizer_compat(tmp_path, tokenizer_grew=False)

    def test_partial_embedding_list_counts_as_having_embeddings(self, tmp_path: Path) -> None:
        """Only `embed_tokens` (no `lm_head`) still triggers the has-embeddings branch."""
        self._write_adapter_config(tmp_path, modules_to_save=["embed_tokens"])
        verify_resume_tokenizer_compat(tmp_path, tokenizer_grew=True)
        with pytest.raises(ResumeIntegrityError):
            verify_resume_tokenizer_compat(tmp_path, tokenizer_grew=False)
