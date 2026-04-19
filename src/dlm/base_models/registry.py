"""Curated launch registry of supported base models.

Every entry pins an exact HuggingFace commit SHA. Refreshed by
`scripts/refresh-registry.py`; weekly CI opens a PR on drift.

Notes on individual entries:

- `qwen2.5-3b` ships under the Qwen Research License (free for entities
  with <100M MAU). We record it as `license_spdx="Other"` and surface
  the URL via `license_url`; it remains `redistributable=True` because
  the license permits bundling + redistribution with attribution. Users
  at scale should read the license.
- Llama-3.2 models are gated on HuggingFace (`requires_acceptance=True`)
  and their license does NOT permit bundling into a `.dlm.pack`
  (`redistributable=False`) — enforced by Sprint 14's pack gate and
  Sprint 28's share-protocol refusal.
- SmolLM2 and Phi-3.5-mini are permissive (Apache-2.0 / MIT).
- `size_gb_fp16` is approximate; the hardware doctor uses it to seed
  VRAM estimates, which then get refined by sprint 09's runtime guard.
"""

from __future__ import annotations

from typing import Final

from dlm.base_models.schema import BaseModelSpec

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
)


BASE_MODELS: Final[dict[str, BaseModelSpec]] = {entry.key: entry for entry in _ENTRIES}


def known_keys() -> tuple[str, ...]:
    """Stable ordering for use in error messages / CLI listings."""
    return tuple(BASE_MODELS.keys())
