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


# --- Sprint 35.3: registry expansion to Qwen2-VL + InternVL2 ----------------

_VL_BASE_KEYS: tuple[str, ...] = (
    "paligemma-3b-mix-224",
    "qwen2-vl-2b-instruct",
    "internvl2-2b",
)


class TestAllVlBasesShipModalityInvariants:
    """Every registered VL base carries the modality + preprocessor plan."""

    @pytest.mark.parametrize("key", _VL_BASE_KEYS)
    def test_modality_is_vl(self, key: str) -> None:
        assert BASE_MODELS[key].modality == "vision-language"

    @pytest.mark.parametrize("key", _VL_BASE_KEYS)
    def test_has_preprocessor_plan(self, key: str) -> None:
        spec = BASE_MODELS[key]
        assert spec.vl_preprocessor_plan is not None
        # Pinned identity fields — each one is part of the cache key,
        # so a silent default would silently invalidate caches.
        assert spec.vl_preprocessor_plan.resize_policy == "fixed"
        assert spec.vl_preprocessor_plan.num_image_tokens > 0

    @pytest.mark.parametrize("key", _VL_BASE_KEYS)
    def test_run_all_drops_export_probes(self, key: str) -> None:
        report = run_all(BASE_MODELS[key])
        names = {r.name for r in report.results}
        # VL bases bypass the three llama.cpp-converter probes.
        assert names.isdisjoint({"gguf_arch", "pretokenizer_label", "pretokenizer_hash"})
        # chat_template probe doesn't apply to VL bases (their chat
        # templates live in the processor, not the tokenizer).
        assert "chat_template" not in names
        assert "vl_image_token" in names
        assert "architecture" in names


class TestQwen2VlRegistryEntry:
    """Sprint 35.3: Qwen2-VL-2B-Instruct landed with pinned fixed res."""

    def test_entry_present(self) -> None:
        assert "qwen2-vl-2b-instruct" in BASE_MODELS

    def test_apache_permissive(self) -> None:
        spec = BASE_MODELS["qwen2-vl-2b-instruct"]
        assert spec.license_spdx == "Apache-2.0"
        assert spec.requires_acceptance is False
        assert spec.redistributable is True

    def test_pinned_preprocessing_plan(self) -> None:
        spec = BASE_MODELS["qwen2-vl-2b-instruct"]
        plan = spec.vl_preprocessor_plan
        assert plan is not None
        # 672×672 with Qwen2-VL's 28-pixel patch-merger grid → 24×24 = 576.
        # Sprint 35.3 pins fixed resolution (implementation-note (a));
        # dynamic-resolution support is deferred to a follow-up.
        assert plan.target_size == (672, 672)
        assert plan.image_token == "<|image_pad|>"
        assert plan.num_image_tokens == 576

    def test_architecture_is_qwen2vl_conditional_generation(self) -> None:
        assert BASE_MODELS["qwen2-vl-2b-instruct"].architecture == "Qwen2VLForConditionalGeneration"

    def test_template_dialect(self) -> None:
        assert BASE_MODELS["qwen2-vl-2b-instruct"].template == "qwen2-vl"


class TestInternVL2RegistryEntry:
    """Sprint 35.3: InternVL2-2B landed with MIT license + 448×448 plan.

    Loader caveat (documented on the registry entry): InternVL2's HF
    integration uses `InternVLChatModel` via remote code — this test
    verifies only the metadata contract, not the runtime load.
    """

    def test_entry_present(self) -> None:
        assert "internvl2-2b" in BASE_MODELS

    def test_mit_permissive(self) -> None:
        spec = BASE_MODELS["internvl2-2b"]
        assert spec.license_spdx == "MIT"
        assert spec.requires_acceptance is False
        assert spec.redistributable is True

    def test_pinned_preprocessing_plan(self) -> None:
        spec = BASE_MODELS["internvl2-2b"]
        plan = spec.vl_preprocessor_plan
        assert plan is not None
        # 448×448 with InternVL2's ViT-L/14 + 2×2 pixel shuffle → 256 tokens.
        assert plan.target_size == (448, 448)
        assert plan.image_token == "<IMG_CONTEXT>"
        assert plan.num_image_tokens == 256

    def test_architecture_is_internvl_chat(self) -> None:
        assert BASE_MODELS["internvl2-2b"].architecture == "InternVLChatModel"

    def test_template_dialect(self) -> None:
        assert BASE_MODELS["internvl2-2b"].template == "internvl2"


class TestDistinctVlBases:
    """The three VL bases occupy distinct rows — no silent duplicates."""

    def test_all_keys_unique(self) -> None:
        assert len(set(_VL_BASE_KEYS)) == 3

    def test_hf_ids_distinct(self) -> None:
        hf_ids = {BASE_MODELS[k].hf_id for k in _VL_BASE_KEYS}
        assert len(hf_ids) == 3

    def test_image_tokens_distinct_per_base(self) -> None:
        """Each VL base uses its native image-token string.

        Silently sharing a placeholder across bases would break the
        cache-key invariant in vl_cache.py (cache key includes the
        token via processor_sha256).
        """
        tokens = {
            BASE_MODELS[k].vl_preprocessor_plan.image_token  # type: ignore[union-attr]
            for k in _VL_BASE_KEYS
        }
        assert len(tokens) == 3


class TestCountVlRegistryEntries:
    """Defend against accidental regression to the PaliGemma-only state."""

    def test_at_least_three_vl_bases_registered(self) -> None:
        vl_count = sum(1 for s in BASE_MODELS.values() if s.modality == "vision-language")
        assert vl_count >= 3


class TestTrustRemoteCodeOptIn:
    """Per-spec trust_remote_code gate (Sprint 35.3 deferred-item follow-up).

    Picking a base that declares `trust_remote_code=True` is the user's
    informed opt-in; the field is False by default so non-custom bases
    never accidentally enable remote code execution at load time.
    """

    def test_defaults_to_false(self) -> None:
        """PaliGemma + Qwen2-VL use standard AutoModel classes."""
        assert BASE_MODELS["paligemma-3b-mix-224"].trust_remote_code is False
        assert BASE_MODELS["qwen2-vl-2b-instruct"].trust_remote_code is False

    def test_internvl2_opts_in(self) -> None:
        """InternVL2 requires trust_remote_code because InternVLChatModel
        is defined in the model repo, not in transformers."""
        assert BASE_MODELS["internvl2-2b"].trust_remote_code is True

    def test_text_bases_default_false(self) -> None:
        """None of the text bases opt into trust_remote_code."""
        for key, spec in BASE_MODELS.items():
            if spec.modality == "text":
                assert spec.trust_remote_code is False, (
                    f"{key} unexpectedly opts into trust_remote_code"
                )
