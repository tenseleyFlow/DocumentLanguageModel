"""VL loader dispatch (Sprint 35 v1).

Covers:

- `load_base_model` picks `AutoModelForImageTextToText` for VL specs
  and `AutoModelForCausalLM` otherwise.
- `load_processor` refuses text-modality specs at the boundary, before
  any transformers import.
- `load_processor` routes VL specs through `AutoProcessor.from_pretrained`.

The real load paths are `pragma: no cover` (they need actual HF models);
these tests patch the transformers AutoClasses so we can assert
dispatch without downloading weights.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dlm.base_models.schema import BaseModelSpec, VlPreprocessorPlan
from dlm.train import loader


def _text_spec() -> BaseModelSpec:
    return BaseModelSpec(
        key="test-text",
        hf_id="test/text",
        revision="a" * 40,
        architecture="LlamaForCausalLM",
        params=1_000_000,
        target_modules=["q_proj"],
        template="chatml",
        gguf_arch="llama",
        tokenizer_pre="llama-bpe",
        license_spdx="Apache-2.0",
        redistributable=True,
        size_gb_fp16=0.5,
        context_length=4096,
        recommended_seq_len=1024,
    )


def _vl_spec() -> BaseModelSpec:
    return BaseModelSpec(
        key="test-vl",
        hf_id="test/vl",
        revision="b" * 40,
        architecture="PaliGemmaForConditionalGeneration",
        params=3_000_000_000,
        target_modules=["q_proj"],
        template="paligemma",
        gguf_arch="paligemma",
        tokenizer_pre="gemma",
        license_spdx="Other",
        redistributable=False,
        size_gb_fp16=6.0,
        context_length=8192,
        recommended_seq_len=2048,
        modality="vision-language",
        vl_preprocessor_plan=VlPreprocessorPlan(
            target_size=(224, 224),
            image_token="<image>",
            num_image_tokens=256,
        ),
    )


def _plan(use_qlora: bool = False) -> Any:
    """Minimal TrainingPlan stub — only fields the loader reads."""
    plan = MagicMock()
    plan.precision = "fp32"
    plan.attn_implementation = "eager"
    plan.use_qlora = use_qlora
    plan.quant_compute_dtype = None
    return plan


class TestLoadBaseModelDispatch:
    def test_text_spec_uses_causal_lm(self) -> None:
        sentinel = MagicMock(name="causal_lm_instance")
        with patch("transformers.AutoModelForCausalLM") as causal:
            causal.from_pretrained.return_value = sentinel
            result = loader.load_base_model(_text_spec(), _plan())
        assert result is sentinel
        causal.from_pretrained.assert_called_once()
        called_hf_id = causal.from_pretrained.call_args.args[0]
        assert called_hf_id == "test/text"
        kwargs = causal.from_pretrained.call_args.kwargs
        assert "dtype" in kwargs
        assert "torch_dtype" not in kwargs

    def test_vl_spec_uses_image_text_to_text(self) -> None:
        sentinel = MagicMock(name="vl_instance")
        with patch("transformers.AutoModelForImageTextToText") as vl:
            vl.from_pretrained.return_value = sentinel
            result = loader.load_base_model(_vl_spec(), _plan())
        assert result is sentinel
        vl.from_pretrained.assert_called_once()
        called_hf_id = vl.from_pretrained.call_args.args[0]
        assert called_hf_id == "test/vl"

    def test_vl_spec_does_not_touch_causal_lm(self) -> None:
        sentinel = MagicMock(name="vl_instance")
        with (
            patch("transformers.AutoModelForImageTextToText") as vl,
            patch("transformers.AutoModelForCausalLM") as causal,
        ):
            vl.from_pretrained.return_value = sentinel
            loader.load_base_model(_vl_spec(), _plan())
        causal.from_pretrained.assert_not_called()


class TestLoadProcessor:
    def test_text_spec_rejected_fast(self) -> None:
        with pytest.raises(ValueError, match="modality='text'"):
            loader.load_processor(_text_spec())

    def test_vl_spec_routes_through_auto_processor(self) -> None:
        # Force-materialize the attribute on the transformers module (HF
        # lazy-loads auto-classes, and `patch` against a lazy name can
        # miss the real accessor). Once the attribute is concrete, patching
        # `from_pretrained` alone is enough to intercept the download.
        import transformers  # type: ignore[import-not-found]

        _ = transformers.AutoProcessor  # noqa: B018 — materializes the attr
        sentinel = MagicMock(name="processor_instance")
        with patch.object(
            transformers.AutoProcessor,
            "from_pretrained",
            return_value=sentinel,
        ) as from_pretrained:
            result = loader.load_processor(_vl_spec())
        assert result is sentinel
        from_pretrained.assert_called_once_with("test/vl", revision="b" * 40)
