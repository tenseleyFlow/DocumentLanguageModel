"""Sprint 40 registry refresh — first 2026 text-family entries."""

from __future__ import annotations

import pytest

from dlm.base_models import BASE_MODELS

_QWEN3_KEYS = ("qwen3-1.7b", "qwen3-4b", "qwen3-8b")


class TestQwen3RegistryEntries:
    @pytest.mark.parametrize("key", _QWEN3_KEYS)
    def test_entry_present(self, key: str) -> None:
        assert key in BASE_MODELS

    @pytest.mark.parametrize("key", _QWEN3_KEYS)
    def test_qwen3_family_uses_expected_architecture(self, key: str) -> None:
        spec = BASE_MODELS[key]
        assert spec.architecture == "Qwen3ForCausalLM"
        assert spec.template == "chatml"
        assert spec.gguf_arch == "qwen3"
        assert spec.tokenizer_pre == "qwen2"

    @pytest.mark.parametrize("key", _QWEN3_KEYS)
    def test_qwen3_family_is_open_and_reasoning_tuned(self, key: str) -> None:
        spec = BASE_MODELS[key]
        assert spec.license_spdx == "Apache-2.0"
        assert spec.requires_acceptance is False
        assert spec.redistributable is True
        assert spec.reasoning_tuned is True
        assert spec.suggested_prompt_temperature == pytest.approx(0.6)

    def test_qwen3_8b_uses_upper_bound_size_hint(self) -> None:
        spec = BASE_MODELS["qwen3-8b"]
        assert spec.params == 8_000_000_000
        assert spec.size_gb_fp16 == pytest.approx(16.0)


class TestLlama33RegistryEntry:
    def test_entry_present(self) -> None:
        assert "llama-3.3-8b-instruct" in BASE_MODELS

    def test_follows_existing_llama_gating_pattern(self) -> None:
        spec = BASE_MODELS["llama-3.3-8b-instruct"]
        assert spec.architecture == "LlamaForCausalLM"
        assert spec.template == "llama3"
        assert spec.gguf_arch == "llama"
        assert spec.requires_acceptance is True
        assert spec.redistributable is False
        assert spec.license_spdx == "Other"

    def test_effective_context_hint_is_lower_than_nominal(self) -> None:
        spec = BASE_MODELS["llama-3.3-8b-instruct"]
        assert spec.context_length == 131_072
        assert spec.context_length_effective == 8_192
        assert spec.effective_context_length == 8_192

    def test_license_url_uses_current_meta_landing_page(self) -> None:
        spec = BASE_MODELS["llama-3.3-8b-instruct"]
        assert spec.license_url == "https://llama.meta.com/llama3/license"


class TestSmolLM3RegistryEntry:
    def test_entry_present(self) -> None:
        assert "smollm3-3b" in BASE_MODELS

    def test_live_hf_id_and_architecture_match_current_release(self) -> None:
        spec = BASE_MODELS["smollm3-3b"]
        assert spec.hf_id == "HuggingFaceTB/SmolLM3-3B"
        assert spec.architecture == "SmolLM3ForCausalLM"
        assert spec.template == "smollm3"
        assert spec.gguf_arch == "llama"
        assert spec.tokenizer_pre == "smollm"

    def test_entry_tracks_reasoning_first_defaults(self) -> None:
        spec = BASE_MODELS["smollm3-3b"]
        assert spec.reasoning_tuned is True
        assert spec.suggested_prompt_temperature == pytest.approx(0.6)

    def test_entry_is_open_and_long_context(self) -> None:
        spec = BASE_MODELS["smollm3-3b"]
        assert spec.license_spdx == "Apache-2.0"
        assert spec.requires_acceptance is False
        assert spec.redistributable is True
        assert spec.context_length == 65_536
        assert spec.recommended_seq_len == 4096


class TestOlmo2RegistryEntry:
    def test_entry_present(self) -> None:
        assert "olmo-2-7b-instruct" in BASE_MODELS

    def test_live_hf_id_and_architecture_match_current_release(self) -> None:
        spec = BASE_MODELS["olmo-2-7b-instruct"]
        assert spec.hf_id == "allenai/OLMo-2-1124-7B-Instruct"
        assert spec.architecture == "Olmo2ForCausalLM"
        assert spec.template == "olmo2"
        assert spec.gguf_arch == "olmo2"
        assert spec.tokenizer_pre == "superbpe"

    def test_entry_is_open_without_reasoning_override(self) -> None:
        spec = BASE_MODELS["olmo-2-7b-instruct"]
        assert spec.license_spdx == "Apache-2.0"
        assert spec.requires_acceptance is False
        assert spec.redistributable is True
        assert spec.reasoning_tuned is False

    def test_entry_uses_model_card_context_and_size_hints(self) -> None:
        spec = BASE_MODELS["olmo-2-7b-instruct"]
        assert spec.params == 7_000_000_000
        assert spec.size_gb_fp16 == pytest.approx(14.6)
        assert spec.context_length == 4096
        assert spec.recommended_seq_len == 2048


class TestGemma2RegistryEntry:
    @pytest.mark.parametrize("key", ["gemma-2-2b-it", "gemma-2-9b-it"])
    def test_entry_present(self, key: str) -> None:
        assert key in BASE_MODELS

    @pytest.mark.parametrize(
        ("key", "hf_id", "params", "size_gb_fp16"),
        [
            ("gemma-2-2b-it", "google/gemma-2-2b-it", 2_600_000_000, 5.2),
            ("gemma-2-9b-it", "google/gemma-2-9b-it", 9_000_000_000, 18.0),
        ],
    )
    def test_live_hf_id_and_architecture_match_current_release(
        self,
        key: str,
        hf_id: str,
        params: int,
        size_gb_fp16: float,
    ) -> None:
        spec = BASE_MODELS[key]
        assert spec.hf_id == hf_id
        assert spec.architecture == "Gemma2ForCausalLM"
        assert spec.template == "gemma2"
        assert spec.gguf_arch == "gemma2"
        assert spec.tokenizer_pre == "gemma"
        assert spec.params == params
        assert spec.size_gb_fp16 == pytest.approx(size_gb_fp16)

    @pytest.mark.parametrize("key", ["gemma-2-2b-it", "gemma-2-9b-it"])
    def test_entry_follows_gemma_gating_pattern(self, key: str) -> None:
        spec = BASE_MODELS[key]
        assert spec.license_spdx == "Gemma"
        assert spec.license_url == "https://ai.google.dev/gemma/terms"
        assert spec.requires_acceptance is True
        assert spec.redistributable is False

    @pytest.mark.parametrize("key", ["gemma-2-2b-it", "gemma-2-9b-it"])
    def test_entry_uses_model_card_context_hints(self, key: str) -> None:
        spec = BASE_MODELS[key]
        assert spec.context_length == 8192
        assert spec.recommended_seq_len == 2048


class TestPhi4MiniReasoningRegistryEntry:
    def test_entry_present(self) -> None:
        assert "phi-4-mini-reasoning" in BASE_MODELS

    def test_live_hf_id_and_architecture_match_current_release(self) -> None:
        spec = BASE_MODELS["phi-4-mini-reasoning"]
        assert spec.hf_id == "microsoft/Phi-4-mini-reasoning"
        assert spec.architecture == "Phi3ForCausalLM"
        assert spec.template == "phi4mini"
        assert spec.gguf_arch == "phi3"
        assert spec.tokenizer_pre == "phi-2"

    def test_entry_is_reasoning_tuned_and_open(self) -> None:
        spec = BASE_MODELS["phi-4-mini-reasoning"]
        assert spec.license_spdx == "MIT"
        assert spec.requires_acceptance is False
        assert spec.redistributable is True
        assert spec.reasoning_tuned is True
        assert spec.suggested_prompt_temperature == pytest.approx(0.6)

    def test_entry_uses_model_card_context_and_size_hints(self) -> None:
        spec = BASE_MODELS["phi-4-mini-reasoning"]
        assert spec.params == 3_800_000_000
        assert spec.size_gb_fp16 == pytest.approx(7.6)
        assert spec.context_length == 131_072
        assert spec.recommended_seq_len == 2048


class TestMixtralRegistryEntry:
    def test_entry_present(self) -> None:
        assert "mixtral-8x7b-instruct" in BASE_MODELS

    def test_entry_is_open_sparse_moe(self) -> None:
        spec = BASE_MODELS["mixtral-8x7b-instruct"]
        assert spec.license_spdx == "Apache-2.0"
        assert spec.requires_acceptance is False
        assert spec.redistributable is True
        assert spec.modality == "text-moe"

    def test_entry_matches_live_family_shape(self) -> None:
        spec = BASE_MODELS["mixtral-8x7b-instruct"]
        assert spec.hf_id == "mistralai/Mixtral-8x7B-Instruct-v0.1"
        assert spec.architecture == "MixtralForCausalLM"
        assert spec.template == "mistral"
        assert spec.gguf_arch == "llama"
        assert spec.tokenizer_pre == "llama-bpe"

    def test_entry_uses_total_parameter_and_context_hints(self) -> None:
        spec = BASE_MODELS["mixtral-8x7b-instruct"]
        assert spec.params == 46_700_000_000
        assert spec.size_gb_fp16 == pytest.approx(93.4)
        assert spec.context_length == 32_768
        assert spec.recommended_seq_len == 2048


class TestStaleSprintDraftRows:
    def test_qwen3_thinking_is_not_a_separate_registry_row(self) -> None:
        """Upstream Qwen3-1.7B ships hybrid thinking in one model.

        Sprint 40's draft listed a separate `qwen3-1.7b-thinking`
        entry, but the live upstream contract exposes thinking mode as
        a switch on `Qwen/Qwen3-1.7B` itself. Keep the registry honest:
        reasoning defaults belong on the real base row, not a fake key.
        """
        assert "qwen3-1.7b-thinking" not in BASE_MODELS

    def test_internvl3_not_shipped_until_remote_code_contract_is_pinned(self) -> None:
        """Guard against copying the stale sprint draft into the registry.

        The live `OpenGVLab/InternVL3-2B` model card still documents
        `trust_remote_code=True`, and on the current stack the whole
        InternVL family still exposes a tokenizer-only `AutoProcessor`
        rather than a complete image processor. Upstream also expands
        `<image>` into repeated `<IMG_CONTEXT>` spans and threads
        `image_flags` through the forward pass. Adding InternVL3 later
        is fine, but it needs an honest runtime contract instead of
        assuming the old "cleaner than InternVL2" sprint note is still
        true.
        """
        assert "internvl3-2b" not in BASE_MODELS
