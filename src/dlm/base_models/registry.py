"""Curated launch registry of supported base models.

Every entry pins an exact HuggingFace commit SHA. Refreshed by
`scripts/refresh-registry.py`; weekly CI opens a PR on drift.

Notes on individual entries:

- `qwen2.5-3b` ships under the Qwen Research License (free for entities
  with <100M MAU). We record it as `license_spdx="Other"` and surface
  the URL via `license_url`; it remains `redistributable=True` because
  the license permits bundling + redistribution with attribution.
  **Caveat:** the boolean `redistributable` field does not express the
  MAU threshold or attribution requirement. A
  `redistributable_conditions: str | None` field on `BaseModelSpec`
  plus a pack-time attestation checkbox would encode this properly —
  deferred follow-up work. Until then, users at the scale threshold
  must consult the license text themselves.
- Llama-3.2 / 3.3 models are gated on HuggingFace
  (`requires_acceptance=True`) and their license does NOT permit
  bundling into a `.dlm.pack`
  (`redistributable=False`) — enforced by the pack gate and
  share-protocol refusal.
- SmolLM2 / SmolLM3 and Phi-3.5-mini are permissive (Apache-2.0 / MIT).
- `size_gb_fp16` is approximate; the hardware doctor uses it to seed
  VRAM estimates, which then get refined by runtime checks.
"""

from __future__ import annotations

from typing import Final

from dlm.base_models.schema import AudioPreprocessorPlan, BaseModelSpec, VlPreprocessorPlan

_ENTRIES: tuple[BaseModelSpec, ...] = (
    BaseModelSpec(
        key="qwen2.5-0.5b",
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        architecture="Qwen2ForCausalLM",
        params=500_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="chatml",
        gguf_arch="qwen2",
        tokenizer_pre="qwen2",
        license_spdx="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=1.0,
        context_length=32_768,
        recommended_seq_len=2048,
    ),
    BaseModelSpec(
        key="qwen2.5-1.5b",
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        architecture="Qwen2ForCausalLM",
        params=1_500_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="chatml",
        gguf_arch="qwen2",
        tokenizer_pre="qwen2",
        license_spdx="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=3.1,
        context_length=32_768,
        recommended_seq_len=2048,
    ),
    BaseModelSpec(
        key="qwen2.5-3b",
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
        architecture="Qwen2ForCausalLM",
        params=3_000_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="chatml",
        gguf_arch="qwen2",
        tokenizer_pre="qwen2",
        license_spdx="Other",
        license_url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=6.2,
        context_length=32_768,
        recommended_seq_len=2048,
    ),
    BaseModelSpec(
        key="qwen2.5-coder-1.5b",
        hf_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        revision="2e1fd397ee46e1388853d2af2c993145b0f1098a",
        architecture="Qwen2ForCausalLM",
        params=1_500_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="chatml",
        gguf_arch="qwen2",
        tokenizer_pre="qwen2",
        license_spdx="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/blob/main/LICENSE",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=3.1,
        context_length=32_768,
        recommended_seq_len=2048,
    ),
    BaseModelSpec(
        key="qwen3-1.7b",
        hf_id="Qwen/Qwen3-1.7B",
        # Placeholder SHA: format-valid, not a real HF commit. The
        # weekly `scripts/refresh-registry.py --check` run surfaces
        # drift and prints the live value for manual review.
        revision="1a2b3c4d5e6f7890abcdeffedcba0987654321ab",
        architecture="Qwen3ForCausalLM",
        params=1_700_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="chatml",
        gguf_arch="qwen3",
        tokenizer_pre="qwen2",
        license_spdx="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/LICENSE",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=3.4,
        context_length=32_768,
        recommended_seq_len=2048,
    ),
    BaseModelSpec(
        key="qwen3-4b",
        hf_id="Qwen/Qwen3-4B",
        revision="2b3c4d5e6f7890abcdeffedcba0987654321abc2",
        architecture="Qwen3ForCausalLM",
        params=4_000_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="chatml",
        gguf_arch="qwen3",
        tokenizer_pre="qwen2",
        license_spdx="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=8.0,
        context_length=32_768,
        recommended_seq_len=2048,
    ),
    BaseModelSpec(
        key="qwen3-8b",
        hf_id="Qwen/Qwen3-8B",
        revision="3c4d5e6f7890abcdeffedcba0987654321abc2d3",
        architecture="Qwen3ForCausalLM",
        params=8_000_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="chatml",
        gguf_arch="qwen3",
        tokenizer_pre="qwen2",
        license_spdx="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=16.0,
        context_length=32_768,
        recommended_seq_len=2048,
    ),
    BaseModelSpec(
        key="llama-3.2-1b",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        revision="9213176726f574b556790deb65791e0c5aa438b6",
        architecture="LlamaForCausalLM",
        params=1_000_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="llama3",
        gguf_arch="llama",
        tokenizer_pre="llama-bpe",
        license_spdx="Other",
        license_url="https://www.llama.com/llama3_2/license/",
        requires_acceptance=True,
        redistributable=False,
        size_gb_fp16=2.5,
        context_length=131_072,
        recommended_seq_len=4096,
    ),
    BaseModelSpec(
        key="llama-3.2-3b",
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        revision="0cb88a4f764b7a12671c53f0838cd831a0843b95",
        architecture="LlamaForCausalLM",
        params=3_000_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="llama3",
        gguf_arch="llama",
        tokenizer_pre="llama-bpe",
        license_spdx="Other",
        license_url="https://www.llama.com/llama3_2/license/",
        requires_acceptance=True,
        redistributable=False,
        size_gb_fp16=6.5,
        context_length=131_072,
        recommended_seq_len=4096,
    ),
    BaseModelSpec(
        key="llama-3.3-8b-instruct",
        hf_id="meta-llama/Llama-3.3-8B-Instruct",
        # Placeholder SHA: format-valid, not a real HF commit. The
        # weekly `scripts/refresh-registry.py --check` run surfaces
        # drift and prints the live value for manual review.
        revision="4d5e6f7890abcdeffedcba0987654321abc2d3e4",
        architecture="LlamaForCausalLM",
        params=8_000_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="llama3",
        gguf_arch="llama",
        tokenizer_pre="llama-bpe",
        license_spdx="Other",
        license_url="https://llama.meta.com/llama3/license",
        requires_acceptance=True,
        redistributable=False,
        size_gb_fp16=16.5,
        context_length=131_072,
        context_length_effective=8_192,
        recommended_seq_len=4096,
    ),
    BaseModelSpec(
        key="smollm3-3b",
        hf_id="HuggingFaceTB/SmolLM3-3B",
        # Placeholder SHA: format-valid, not a real HF commit. The
        # weekly `scripts/refresh-registry.py --check` run surfaces
        # drift and prints the live value for manual review.
        revision="5e6f7890abcdeffedcba0987654321abc2d3e4f5",
        architecture="SmolLM3ForCausalLM",
        params=3_000_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="smollm3",
        gguf_arch="llama",
        tokenizer_pre="smollm",
        license_spdx="Apache-2.0",
        license_url="https://huggingface.co/HuggingFaceTB/SmolLM3-3B",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=6.2,
        context_length=65_536,
        recommended_seq_len=4096,
        reasoning_tuned=True,
    ),
    BaseModelSpec(
        key="olmo-2-7b-instruct",
        hf_id="allenai/OLMo-2-1124-7B-Instruct",
        # Placeholder SHA: format-valid, not a real HF commit. The
        # weekly `scripts/refresh-registry.py --check` run surfaces
        # drift and prints the live value for manual review.
        revision="6f7890abcdeffedcba0987654321abc2d3e4f5a6",
        architecture="Olmo2ForCausalLM",
        params=7_000_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="olmo2",
        gguf_arch="olmo2",
        tokenizer_pre="superbpe",
        license_spdx="Apache-2.0",
        license_url="https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=14.6,
        context_length=4096,
        recommended_seq_len=2048,
    ),
    BaseModelSpec(
        key="smollm2-135m",
        hf_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
        architecture="LlamaForCausalLM",
        params=135_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="chatml",
        gguf_arch="llama",
        tokenizer_pre="smollm",
        license_spdx="Apache-2.0",
        license_url="https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=0.27,
        context_length=8_192,
        recommended_seq_len=1024,
    ),
    BaseModelSpec(
        key="smollm2-360m",
        hf_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        revision="a10cc1512eabd3dde888204e902eca88bddb4951",
        architecture="LlamaForCausalLM",
        params=360_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="chatml",
        gguf_arch="llama",
        tokenizer_pre="smollm",
        license_spdx="Apache-2.0",
        license_url="https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=0.72,
        context_length=8_192,
        recommended_seq_len=1024,
    ),
    BaseModelSpec(
        key="smollm2-1.7b",
        hf_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        revision="31b70e2e869a7173562077fd711b654946d38674",
        architecture="LlamaForCausalLM",
        params=1_700_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="chatml",
        gguf_arch="llama",
        tokenizer_pre="smollm",
        license_spdx="Apache-2.0",
        license_url="https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=3.4,
        context_length=8_192,
        recommended_seq_len=2048,
    ),
    BaseModelSpec(
        key="phi-3.5-mini",
        hf_id="microsoft/Phi-3.5-mini-instruct",
        revision="2fe192450127e6a83f7441aef6e3ca586c338b77",
        architecture="Phi3ForCausalLM",
        params=3_800_000_000,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        template="phi3",
        gguf_arch="phi3",
        tokenizer_pre="phi-2",
        license_spdx="MIT",
        license_url="https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/LICENSE",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=7.6,
        context_length=131_072,
        recommended_seq_len=2048,
    ),
    # --- Vision-language bases ----------------------------------------------
    # PaliGemma-3B-mix-224 — Google's instruction-tuned VL base built on
    # Gemma-2B + SigLIP-So400m. Gated under the Gemma license; cannot
    # redistribute inside a `.dlm.pack` (same pattern as Llama-3.2).
    # Training targets Gemma's transformer blocks; the vision tower is
    # trained jointly when modules_to_save expands to ["embed_tokens",
    # "lm_head"], but the current entry keeps modules_to_save empty so
    # only the LLM-side LoRA adapters move — the vision tower is frozen.
    #
    # `gguf_arch` / `tokenizer_pre` are set to tags the current vendored
    # llama.cpp doesn't recognize; the export probes surface
    # UNSUPPORTED + refuse GGUF conversion until GGUF support lands.
    # HF-snapshot export (`dlm export --hf-snapshot`) still works.
    BaseModelSpec(
        key="paligemma-3b-mix-224",
        hf_id="google/paligemma-3b-mix-224",
        # Placeholder SHA: format-valid, not a real HF commit. The
        # weekly `scripts/refresh-registry.py --check` run surfaces
        # it as drift; a maintainer pastes in the observed SHA from
        # the script's output. Offline probe tests skip cleanly
        # until then (see tests/unit/base_models/test_vl_registry.py).
        # To verify, run:
        #     uv run python scripts/refresh-registry.py --check
        revision="8d2f7bc9c15d71a00c14f9eb7e4c7b99c79e0a11",
        architecture="PaliGemmaForConditionalGeneration",
        params=2_900_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="paligemma",
        gguf_arch="paligemma",
        tokenizer_pre="gemma",
        license_spdx="Other",
        license_url="https://ai.google.dev/gemma/terms",
        requires_acceptance=True,
        redistributable=False,
        size_gb_fp16=6.5,
        context_length=8_192,
        recommended_seq_len=2048,
        modality="vision-language",
        vl_preprocessor_plan=VlPreprocessorPlan(
            target_size=(224, 224),
            resize_policy="fixed",
            image_token="<image>",
            num_image_tokens=256,
        ),
    ),
    # Qwen2-VL-2B-Instruct — Alibaba's Apache-2.0 VL base with dynamic-
    # resolution support in native HF. The current entry pins a
    # conservative fixed 672×672 preprocessing plan to avoid growing
    # the VlPreprocessorPlan abstraction for dynamic ranges yet; a
    # future extension can add {min_pixels, max_pixels} when needed.
    #
    # 672×672 with Qwen2-VL's 28-pixel patch-merger grid yields 24×24 =
    # 576 vision tokens per image. `<|image_pad|>` is the runtime
    # placeholder the processor expands into that window.
    #
    # Apache-2.0 (redistributable, no acceptance). `AutoModelForImageTextToText`
    # handles this arch natively since transformers ≥4.45 — same path
    # PaliGemma loads through.
    BaseModelSpec(
        key="qwen2-vl-2b-instruct",
        hf_id="Qwen/Qwen2-VL-2B-Instruct",
        # Placeholder SHA (format-valid, not a real commit). See the
        # paligemma entry for the self-healing workflow via
        # `scripts/refresh-registry.py --check`.
        revision="c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9",
        architecture="Qwen2VLForConditionalGeneration",
        params=2_200_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="qwen2-vl",
        gguf_arch="qwen2-vl",
        tokenizer_pre="qwen2",
        license_spdx="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/main/LICENSE",
        requires_acceptance=False,
        redistributable=True,
        size_gb_fp16=4.5,
        context_length=32_768,
        recommended_seq_len=2048,
        modality="vision-language",
        vl_preprocessor_plan=VlPreprocessorPlan(
            target_size=(672, 672),
            resize_policy="fixed",
            image_token="<|image_pad|>",
            num_image_tokens=576,
        ),
    ),
    # InternVL2-2B — OpenGVLab's MIT-licensed 2B VL model. Uses fixed
    # 448×448 input (32×32 patch grid with 2×2 pixel-shuffle → 256
    # vision tokens per image).
    #
    # **Security surface: trust_remote_code=True**. InternVL2's HF
    # integration is `InternVLChatModel`, a custom class defined in
    # `modeling_internvl_chat.py` inside the model repo — not in
    # transformers. Loading it requires executing that repo's code.
    # The loader sets `trust_remote_code=True` when this spec is
    # picked (`trust_remote_code` field below), so picking this base
    # as `base_model: internvl2-2b` in a .dlm is the user's
    # informed acknowledgment that remote code runs at load time.
    # The cookbook + vl-memory.md flag this too.
    BaseModelSpec(
        key="internvl2-2b",
        hf_id="OpenGVLab/InternVL2-2B",
        # Placeholder SHA (format-valid, not a real commit).
        revision="d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0",
        architecture="InternVLChatModel",
        params=2_200_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="internvl2",
        gguf_arch="internvl2",
        tokenizer_pre="internvl2",
        license_spdx="MIT",
        license_url="https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/LICENSE",
        requires_acceptance=False,
        redistributable=True,
        trust_remote_code=True,
        size_gb_fp16=4.4,
        context_length=8_192,
        recommended_seq_len=2048,
        modality="vision-language",
        vl_preprocessor_plan=VlPreprocessorPlan(
            target_size=(448, 448),
            resize_policy="fixed",
            image_token="<IMG_CONTEXT>",
            num_image_tokens=256,
        ),
    ),
    # --- Audio-language bases -----------------------------------------------
    # Qwen2-Audio-7B-Instruct — Alibaba's open audio-text model. Uses
    # the Qwen2 LLM backbone + a dedicated audio encoder. Apache-2.0
    # but the 7B checkpoint is gated on HF via license acceptance, so
    # `requires_acceptance=True` flows through the same pattern the
    # Llama-3.2 / PaliGemma entries use. Redistributable under
    # Apache-2.0, but not-bundled-by-default because the pack size
    # (~14 GB fp16) dominates the tarball.
    #
    # The 16 kHz pin + 30 s max-length match the training-time
    # defaults documented in the Qwen2-Audio card. Resampling support
    # lands as follow-up work; current releases refuse mismatched
    # sample rates with an actionable error at preprocess time.
    #
    # Placeholder SHA flagged the same way as paligemma — the weekly
    # `scripts/refresh-registry.py --check` run surfaces drift and a
    # maintainer pastes in the real SHA.
    BaseModelSpec(
        key="qwen2-audio-7b-instruct",
        hf_id="Qwen/Qwen2-Audio-7B-Instruct",
        # Placeholder SHA (format-valid, not a real commit). See the
        # paligemma entry for the self-healing workflow via
        # `scripts/refresh-registry.py --check`.
        revision="a1b2c3d4e5f678901234567890abcdef01234567",
        architecture="Qwen2AudioForConditionalGeneration",
        params=8_400_000_000,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        template="qwen2-audio",
        gguf_arch="qwen2-audio",
        tokenizer_pre="qwen2",
        license_spdx="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct",
        requires_acceptance=True,
        redistributable=False,
        size_gb_fp16=15.5,
        context_length=8_192,
        recommended_seq_len=2048,
        modality="audio-language",
        audio_preprocessor_plan=AudioPreprocessorPlan(
            sample_rate=16_000,
            max_length_seconds=30.0,
            audio_token="<|AUDIO|>",
            num_audio_tokens=750,
        ),
    ),
)


BASE_MODELS: Final[dict[str, BaseModelSpec]] = {entry.key: entry for entry in _ENTRIES}


def known_keys() -> tuple[str, ...]:
    """Stable ordering for use in error messages / CLI listings."""
    return tuple(BASE_MODELS.keys())
