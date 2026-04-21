"""VL-base registry + preprocessor plan + VL probe (Sprint 35 v1).

Covers:

- `paligemma-3b-mix-224` is present and has `modality="vision-language"`.
- Its `VlPreprocessorPlan` is pinned (224×224, fixed, `<image>`, 256).
- License is Gemma — gated, not redistributable.
- `modality="vision-language"` without a preprocessor plan rejects at
  validate time; text bases cannot carry a plan.
- `run_all` on a VL spec skips the llama.cpp-converter probes (GGUF
  support for VL archs is Sprint 35.4's scope).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlm.base_models import BASE_MODELS
from dlm.base_models.probes import run_all
from dlm.base_models.schema import BaseModelSpec, VlPreprocessorPlan


class TestPaliGemmaRegistryEntry:
    def test_entry_present(self) -> None:
        assert "paligemma-3b-mix-224" in BASE_MODELS

    def test_modality_is_vl(self) -> None:
        spec = BASE_MODELS["paligemma-3b-mix-224"]
        assert spec.modality == "vision-language"

    def test_preprocessor_plan_pinned(self) -> None:
        spec = BASE_MODELS["paligemma-3b-mix-224"]
        plan = spec.vl_preprocessor_plan
        assert plan is not None
        assert plan.target_size == (224, 224)
        assert plan.resize_policy == "fixed"
        assert plan.image_token == "<image>"
        assert plan.num_image_tokens == 256

    def test_license_gated_not_redistributable(self) -> None:
        spec = BASE_MODELS["paligemma-3b-mix-224"]
        assert spec.requires_acceptance is True
        assert spec.redistributable is False

    def test_architecture_is_vl_conditional_generation(self) -> None:
        spec = BASE_MODELS["paligemma-3b-mix-224"]
        assert spec.architecture == "PaliGemmaForConditionalGeneration"


class TestVlPreprocessorPlan:
    def test_rejects_non_positive_target_size(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            VlPreprocessorPlan(
                target_size=(0, 224),
                image_token="<image>",
                num_image_tokens=256,
            )

    def test_frozen(self) -> None:
        plan = VlPreprocessorPlan(
            target_size=(224, 224),
            image_token="<image>",
            num_image_tokens=256,
        )
        with pytest.raises(ValidationError):
            plan.num_image_tokens = 512  # type: ignore[misc]

    def test_unknown_resize_policy_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VlPreprocessorPlan(
                target_size=(224, 224),
                image_token="<image>",
                num_image_tokens=256,
                resize_policy="quadratic",  # type: ignore[arg-type]
            )


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

    def test_vl_without_plan_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires a vl_preprocessor_plan"):
            BaseModelSpec(**self._base_kwargs(), modality="vision-language")  # type: ignore[arg-type]

    def test_text_with_plan_rejected(self) -> None:
        plan = VlPreprocessorPlan(
            target_size=(224, 224),
            image_token="<image>",
            num_image_tokens=256,
        )
        with pytest.raises(ValidationError, match="only valid with"):
            BaseModelSpec(  # type: ignore[arg-type]
                **self._base_kwargs(),
                modality="text",
                vl_preprocessor_plan=plan,
            )


class TestRunAllSkipsExportProbesForVl:
    """`run_all` on a VL spec auto-drops the three llama.cpp probes.

    GGUF export for VL archs is Sprint 35.4's scope; forcing the probes
    here would either false-fail (converter doesn't know the arch) or
    skip with a noisy per-probe message. The dispatcher quietly omits
    them instead.
    """

    def test_vl_spec_yields_two_probes(self) -> None:
        spec = BASE_MODELS["paligemma-3b-mix-224"]
        report = run_all(spec)
        probe_names = {r.name for r in report.results}
        assert "gguf_arch" not in probe_names
        assert "pretokenizer_label" not in probe_names
        assert "pretokenizer_hash" not in probe_names
        # vl_image_token is the VL-specific probe; it may skip if
        # transformers/processor isn't cached locally.
        assert "vl_image_token" in probe_names
        assert "architecture" in probe_names
        # Chat-template probe does not apply to VL bases.
        assert "chat_template" not in probe_names
