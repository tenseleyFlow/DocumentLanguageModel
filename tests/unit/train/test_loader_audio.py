"""Audio loader dispatch (Sprint 35.2 T8).

Parallel to `test_loader_vl.py`. Covers:

- `load_base_model` picks the architecture-named class for audio specs
  (`Qwen2AudioForConditionalGeneration`).
- `load_processor` now accepts audio-language specs in addition to VL.
- Unknown audio architectures raise a readable error (unmapped class
  in `_AUDIO_MODEL_CLASSES`).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dlm.base_models.schema import AudioPreprocessorPlan, BaseModelSpec
from dlm.train import loader


def _audio_spec(architecture: str = "Qwen2AudioForConditionalGeneration") -> BaseModelSpec:
    return BaseModelSpec(
        key="test-audio",
        hf_id="test/audio",
        revision="c" * 40,
        architecture=architecture,
        params=8_000_000_000,
        target_modules=["q_proj"],
        template="qwen2-audio",
        gguf_arch="qwen2-audio",
        tokenizer_pre="qwen2",
        license_spdx="Apache-2.0",
        redistributable=False,
        size_gb_fp16=15.0,
        context_length=8192,
        recommended_seq_len=2048,
        modality="audio-language",
        audio_preprocessor_plan=AudioPreprocessorPlan(
            sample_rate=16_000,
            max_length_seconds=30.0,
            audio_token="<|AUDIO|>",
            num_audio_tokens=750,
        ),
    )


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


def _plan() -> Any:
    plan = MagicMock()
    plan.precision = "fp32"
    plan.attn_implementation = "eager"
    plan.use_qlora = False
    plan.quant_compute_dtype = None
    return plan


class TestLoadBaseModelAudioDispatch:
    def test_audio_spec_uses_architecture_class(self) -> None:
        sentinel = MagicMock(name="qwen2_audio_instance")
        # HF lazy-loads auto-classes, so we force-materialize the attr
        # first by touching it — that way the patch below hits the
        # resolved class instead of the lazy-loader proxy. Swapping
        # `from_pretrained` only is enough to intercept the download.
        import transformers  # type: ignore[import-not-found]

        real_cls = transformers.Qwen2AudioForConditionalGeneration
        with patch.object(real_cls, "from_pretrained", return_value=sentinel) as from_pretrained:
            result = loader.load_base_model(_audio_spec(), _plan())
        assert result is sentinel
        from_pretrained.assert_called_once()
        assert from_pretrained.call_args.args[0] == "test/audio"

    def test_audio_spec_skips_causal_lm(self) -> None:
        sentinel = MagicMock(name="qwen2_audio_instance")
        import transformers  # type: ignore[import-not-found]

        real_cls = transformers.Qwen2AudioForConditionalGeneration
        with (
            patch.object(real_cls, "from_pretrained", return_value=sentinel),
            patch("transformers.AutoModelForCausalLM") as causal,
            patch("transformers.AutoModelForImageTextToText") as vl,
        ):
            loader.load_base_model(_audio_spec(), _plan())
        causal.from_pretrained.assert_not_called()
        vl.from_pretrained.assert_not_called()

    def test_unknown_audio_architecture_refused(self) -> None:
        spec = _audio_spec(architecture="MysteryAudioModel")
        with pytest.raises(ValueError, match="no audio-LM loader wired for architecture"):
            loader.load_base_model(spec, _plan())


class TestLoadProcessorAudio:
    def test_audio_spec_accepted(self) -> None:
        import transformers  # type: ignore[import-not-found]

        _ = transformers.AutoProcessor  # noqa: B018 — materializes the attr
        sentinel = MagicMock(name="audio_processor")
        with patch.object(
            transformers.AutoProcessor,
            "from_pretrained",
            return_value=sentinel,
        ) as from_pretrained:
            result = loader.load_processor(_audio_spec())
        assert result is sentinel
        from_pretrained.assert_called_once_with("test/audio", revision="c" * 40)

    def test_text_spec_still_rejected(self) -> None:
        with pytest.raises(ValueError, match="modality='text'"):
            loader.load_processor(_text_spec())
