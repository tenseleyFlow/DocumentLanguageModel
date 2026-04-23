"""Audio-base registry + preprocessor plan + audio probe (Sprint 35.2).

Mirrors `test_vl_registry.py` for the audio-language modality. Covers:

- `qwen2-audio-7b-instruct` is present and has `modality="audio-language"`.
- Its `AudioPreprocessorPlan` is pinned (16 kHz, 30 s, `<|AUDIO|>`, 750).
- License is Apache-2.0 and the current HF row is no longer gated, so
  the spec stays redistributable.
- `modality="audio-language"` without a plan rejects at validate time;
  text bases cannot carry an audio plan; VL bases cannot carry an audio
  plan; audio bases cannot carry a VL plan.
- `run_all` on an audio spec skips the llama.cpp-converter probes (no
  audio-arch support on any llama.cpp roadmap).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlm.base_models import BASE_MODELS
from dlm.base_models.probes import run_all
from dlm.base_models.schema import (
    AudioPreprocessorPlan,
    BaseModelSpec,
    VlPreprocessorPlan,
)


class TestQwen2AudioRegistryEntry:
    def test_entry_present(self) -> None:
        assert "qwen2-audio-7b-instruct" in BASE_MODELS

    def test_modality_is_audio(self) -> None:
        spec = BASE_MODELS["qwen2-audio-7b-instruct"]
        assert spec.modality == "audio-language"

    def test_preprocessor_plan_pinned(self) -> None:
        spec = BASE_MODELS["qwen2-audio-7b-instruct"]
        plan = spec.audio_preprocessor_plan
        assert plan is not None
        assert plan.sample_rate == 16_000
        assert plan.max_length_seconds == 30.0
        assert plan.audio_token == "<|AUDIO|>"
        assert plan.num_audio_tokens == 750

    def test_no_vl_plan(self) -> None:
        spec = BASE_MODELS["qwen2-audio-7b-instruct"]
        assert spec.vl_preprocessor_plan is None

    def test_license_open_and_redistributable(self) -> None:
        spec = BASE_MODELS["qwen2-audio-7b-instruct"]
        assert spec.requires_acceptance is False
        assert spec.redistributable is True

    def test_architecture_is_audio_conditional_generation(self) -> None:
        spec = BASE_MODELS["qwen2-audio-7b-instruct"]
        assert spec.architecture == "Qwen2AudioForConditionalGeneration"

    def test_template_is_qwen2_audio(self) -> None:
        spec = BASE_MODELS["qwen2-audio-7b-instruct"]
        assert spec.template == "qwen2-audio"


class TestAudioPreprocessorPlan:
    def test_rejects_non_positive_sample_rate(self) -> None:
        with pytest.raises(ValidationError):
            AudioPreprocessorPlan(
                sample_rate=0,
                max_length_seconds=30.0,
                audio_token="<|AUDIO|>",
                num_audio_tokens=750,
            )

    def test_rejects_non_positive_max_length(self) -> None:
        with pytest.raises(ValidationError):
            AudioPreprocessorPlan(
                sample_rate=16_000,
                max_length_seconds=0.0,
                audio_token="<|AUDIO|>",
                num_audio_tokens=750,
            )

    def test_rejects_empty_audio_token(self) -> None:
        with pytest.raises(ValidationError):
            AudioPreprocessorPlan(
                sample_rate=16_000,
                max_length_seconds=30.0,
                audio_token="",
                num_audio_tokens=750,
            )

    def test_rejects_non_positive_num_audio_tokens(self) -> None:
        with pytest.raises(ValidationError):
            AudioPreprocessorPlan(
                sample_rate=16_000,
                max_length_seconds=30.0,
                audio_token="<|AUDIO|>",
                num_audio_tokens=0,
            )

    def test_frozen(self) -> None:
        plan = AudioPreprocessorPlan(
            sample_rate=16_000,
            max_length_seconds=30.0,
            audio_token="<|AUDIO|>",
            num_audio_tokens=750,
        )
        with pytest.raises(ValidationError):
            plan.num_audio_tokens = 1500  # type: ignore[misc]


class TestSpecModalityInvariants:
    def _base_kwargs(self) -> dict[str, object]:
        return {
            "key": "test-entry",
            "hf_id": "test/entry",
            "revision": "a" * 40,
            "architecture": "LlamaForCausalLM",
            "params": 1_000_000,
            "target_modules": ["q_proj"],
            "template": "chatml",
            "gguf_arch": "llama",
            "tokenizer_pre": "llama-bpe",
            "license_spdx": "Apache-2.0",
            "redistributable": True,
            "size_gb_fp16": 0.5,
            "context_length": 4096,
            "recommended_seq_len": 1024,
        }

    def _audio_plan(self) -> AudioPreprocessorPlan:
        return AudioPreprocessorPlan(
            sample_rate=16_000,
            max_length_seconds=30.0,
            audio_token="<|AUDIO|>",
            num_audio_tokens=750,
        )

    def _vl_plan(self) -> VlPreprocessorPlan:
        return VlPreprocessorPlan(
            target_size=(224, 224),
            image_token="<image>",
            num_image_tokens=256,
        )

    def test_audio_without_plan_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires an audio_preprocessor_plan"):
            BaseModelSpec(**self._base_kwargs(), modality="audio-language")  # type: ignore[arg-type]

    def test_text_with_audio_plan_rejected(self) -> None:
        with pytest.raises(ValidationError, match="only valid with"):
            BaseModelSpec(  # type: ignore[arg-type]
                **self._base_kwargs(),
                modality="text",
                audio_preprocessor_plan=self._audio_plan(),
            )

    def test_vl_with_audio_plan_rejected(self) -> None:
        with pytest.raises(ValidationError, match="audio_preprocessor_plan is invalid"):
            BaseModelSpec(  # type: ignore[arg-type]
                **self._base_kwargs(),
                modality="vision-language",
                vl_preprocessor_plan=self._vl_plan(),
                audio_preprocessor_plan=self._audio_plan(),
            )

    def test_audio_with_vl_plan_rejected(self) -> None:
        with pytest.raises(ValidationError, match="vl_preprocessor_plan is invalid"):
            BaseModelSpec(  # type: ignore[arg-type]
                **self._base_kwargs(),
                modality="audio-language",
                audio_preprocessor_plan=self._audio_plan(),
                vl_preprocessor_plan=self._vl_plan(),
            )


class TestRunAllSkipsExportProbesForAudio:
    """`run_all` on an audio spec drops the llama.cpp-converter probes.

    Audio architectures aren't on any llama.cpp roadmap; GGUF export
    refuses cleanly and emits an HF snapshot instead. The dispatcher
    quietly omits the export probes to keep the report focused.
    """

    def test_audio_spec_yields_two_probes(self) -> None:
        spec = BASE_MODELS["qwen2-audio-7b-instruct"]
        report = run_all(spec)
        probe_names = {r.name for r in report.results}
        assert "gguf_arch" not in probe_names
        assert "pretokenizer_label" not in probe_names
        assert "pretokenizer_hash" not in probe_names
        # audio_token is the audio-specific probe; it may skip if
        # transformers/processor isn't cached locally.
        assert "audio_token" in probe_names
        assert "architecture" in probe_names
        # Chat-template probe does not apply to audio bases.
        assert "chat_template" not in probe_names

    def test_audio_spec_skips_vl_probe(self) -> None:
        spec = BASE_MODELS["qwen2-audio-7b-instruct"]
        report = run_all(spec)
        probe_names = {r.name for r in report.results}
        assert "vl_image_token" not in probe_names
