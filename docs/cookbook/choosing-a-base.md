# Choosing a base

The fastest way to pick a DLM base is to decide three things first:

1. Do you need plain text, multimodal vision, or audio?
2. Do you want the most permissive license possible, or are gated rows fine?
3. Are you targeting Apple Silicon, a mid-size CUDA card, or a large CUDA box?

## Quick picks

| If you want… | Start with… | Why |
|---|---|---|
| Fast local iteration on almost any laptop | `smollm2-135m` | Tiny, cheap, and ideal for testing authoring loops. |
| Best general-purpose 2026 text base around the 4B tier | `qwen3-4b` | Strong default quality, permissive license, and current-generation tokenizer/chat behavior. |
| A reasoning-first 1.7B profile | `qwen3-1.7b-thinking` | Same upstream Qwen3 weights, but a curated reasoning-profile key with cooler defaults. |
| Fully open-model story | `olmo-2-7b-instruct` | Open weights and open-data lineage make it the cleanest reproducibility pitch. |
| Apache sparse-MoE experiments | `mixtral-8x7b-instruct` | First `text-moe` row in the registry; pairs with the learned gate work. |
| Small gated text base | `gemma-2-2b-it` | Useful when Gemma’s instruction style or ecosystem matters more than license friction. |
| Larger gated text base | `gemma-2-9b-it` | Upper-tier Gemma pick; large enough to want real GPU planning. |
| Large multimodal capability | `mistral-small-3.1-24b-instruct` | Strongest shipped VL row, but large-CUDA-first. |
| Safe default multimodal row on a smaller box | `qwen2-vl-2b-instruct` | Permissive, solid, and compatible with the current generic VL runtime. |
| Audio-language training | `qwen2-audio-7b-instruct` | Current shipped audio row; open-license and no longer gated on HF. |

## Notes on the sharp edges

- `llama-3.3-8b-instruct` is still treated like the Llama family in DLM’s policy surface: acceptance required, not redistributable, and intended for users who already know they want the Llama line. Today it resolves through a community HF mirror while DLM pins provenance against Meta’s official LlamaCon/newsroom announcement, because Meta has not published a first-party HF repo for this row.
- `internvl2-2b` and `internvl3-2b` are registry-visible planning targets, but the current generic VL runtime still refuses the InternVL family until DLM owns its custom processor/collator contract.
- `mistral-small-3.1-24b-instruct` is intentionally refused on MPS by default. It is a real shipped row, just not a casual laptop target.

## Hardware-first view

- Apple Silicon, 16 GB: `smollm2-*`, `qwen2.5-*`, `qwen3-1.7b`, and `qwen3-4b` are the comfortable text picks; `qwen2-vl-2b-instruct` is the safer VL row.
- Apple Silicon, 32 GB+: `qwen3-8b`, `gemma-2-2b-it`, and `phi-4-mini-reasoning` become practical. Large VL rows still need caution.
- CUDA, 24 GB: this is where `gemma-2-9b-it`, `mixtral-8x7b-instruct`, and the heavier multimodal rows start becoming realistic.
- CUDA, 48 GB+: this is the intended home for `mistral-small-3.1-24b-instruct`.

See [hardware/memory-estimates](../hardware/memory-estimates.md) for the text-family budget table and [hardware/vl-memory](../hardware/vl-memory.md) for the VL rows.
