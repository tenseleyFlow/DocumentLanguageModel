# Vision-language memory budget

Three VL bases ship after Sprint 35.3: **PaliGemma-3B-mix-224**,
**Qwen2-VL-2B-Instruct**, and **InternVL2-2B**. Each is pinned at a
fixed preprocessing resolution; dynamic-resolution support (Qwen2-VL's
native capability) is deferred to a follow-up so the
`VlPreprocessorPlan` cache key stays stable.

## Base-selection guidance

| Base                      | License    | Pick when you want… |
|---------------------------|------------|---------------------|
| paligemma-3b-mix-224      | Gemma (gated) | The cleanest PEFT path + proven chart/doc QA; accept the Gemma license first. |
| qwen2-vl-2b-instruct      | Apache-2.0 | Permissive licensing + strong general-purpose VL; dynamic-res is capped to 672² in v1 but native runtime supports more. |
| internvl2-2b              | MIT        | Most permissive license + competitive 2B-scale quality; **loader caveat** (InternVLChatModel uses trust_remote_code). |

## PaliGemma-3B-mix-224 (224×224, fp16)

All numbers in GB. "Training" includes the base weights + r=16 LoRA
adapters + optimizer state (AdamW, 2x master copy) + per-batch
activation + gradient checkpointing.

| Config          | Base weights | Adapter | Activations | Total (peak) |
|-----------------|-------------:|--------:|------------:|-------------:|
| Inference, fp16 |          6.5 |    0.04 |         0.4 |          7.0 |
| LoRA + bs=1     |          6.5 |    0.04 |         2.0 |         10.0 |
| LoRA + bs=4     |          6.5 |    0.04 |         8.0 |         16.5 |

**Floor.** MPS with 16 GB unified memory handles inference + LoRA at
batch=1 comfortably; batch=4 overshoots and triggers OOM. Users who
need batch=4+ on Apple Silicon: wait for a 24 GB+ box, or use
gradient accumulation (`training.grad_accum: 4` + `micro_batch_size:
1` gives the same effective batch at LoRA cost).

**CUDA floor.** SM 8.0 with 12 GB VRAM comfortably handles LoRA
batch=1; SM 8.0 with 24 GB handles batch=4 directly. QLoRA on VL
isn't plumbed in v1 (see Sprint 35.3 follow-up).

## Qwen2-VL-2B-Instruct (pinned 672×672, fp16)

Qwen2-VL's HF-native dynamic resolution is capped to a fixed 672²
preprocessing plan in v1 — 24×24 patch grid × patch-merger 2×2 yields
576 image tokens per frame, which is the cache-key invariant.

| Config          | Base weights | Adapter | Activations | Total (peak) |
|-----------------|-------------:|--------:|------------:|-------------:|
| Inference, fp16 |          4.5 |    0.03 |         0.8 |          5.4 |
| LoRA + bs=1     |          4.5 |    0.03 |         3.2 |          7.8 |
| LoRA + bs=4     |          4.5 |    0.03 |        12.8 |         17.4 |

**Floor.** MPS with 16 GB unified memory handles LoRA batch=1 with
headroom for IDE + browser. 24 GB CUDA fits batch=4. Larger images
than 672² inflate activation memory super-linearly (576 tokens grows
as `(H/28) × (W/28)`); revisit when the plan supports dynamic ranges.

**Qwen2-VL-specific.** The vision tower is a 675M-param ViT so the
activation footprint at LoRA time is dominated by cross-attention
between vision + text tokens. Gradient checkpointing on the tower
trims ~30% of peak; `training.gradient_checkpointing: true` in
frontmatter enables it.

## InternVL2-2B (448×448, fp16)

InternVL2 uses ViT-L/14 + pixel-shuffle 2×2 so 448² input yields 256
image tokens — the smallest of the three bases and cheapest at
training time.

| Config          | Base weights | Adapter | Activations | Total (peak) |
|-----------------|-------------:|--------:|------------:|-------------:|
| Inference, fp16 |          4.4 |    0.03 |         0.3 |          4.8 |
| LoRA + bs=1     |          4.4 |    0.03 |         1.5 |          6.0 |
| LoRA + bs=4     |          4.4 |    0.03 |         6.0 |         10.5 |

**Floor.** MPS with 16 GB comfortably handles batch=4. 12 GB CUDA
handles batch=1; 16 GB CUDA handles batch=4.

**Security note: trust_remote_code.** InternVL2 ships as
`InternVLChatModel`, a custom class defined in
`modeling_internvl_chat.py` inside the HF model repo. Loading it
requires executing that repo's code — the registry entry declares
`trust_remote_code=True`, and the loader routes through
`AutoModel.from_pretrained(trust_remote_code=True)`. Picking this
base in a `.dlm` frontmatter is the user's informed acknowledgment:
the other two VL bases ship their class in transformers itself and
do NOT set `trust_remote_code`.

## llama.cpp GGUF support matrix (sprint 35.4)

`dlm.export.arch_probe` scans the vendored `convert_hf_to_gguf.py`
for each VL arch and classifies coverage. Current verdicts at tag
**b8816** (cached in `vendor/llama_cpp_vl_arch_support.json`, refreshed
by `scripts/bump-llama-cpp.sh bump <tag>`):

| Base                      | Arch class                          | GGUF support |
|---------------------------|-------------------------------------|:-------------|
| paligemma-3b-mix-224      | PaliGemmaForConditionalGeneration   | UNSUPPORTED  |
| qwen2-vl-2b-instruct      | Qwen2VLForConditionalGeneration     | SUPPORTED    |
| internvl2-2b              | InternVLChatModel                   | UNSUPPORTED  |

**UNSUPPORTED** means `dlm export` falls back to the HF-snapshot path
with an actionable banner. **SUPPORTED** means single-file VL GGUF
emission runs: `dlm export --merged --quant Q4_K_M` orchestrates merge
→ `convert_hf_to_gguf.py` → `llama-quantize` → render a Modelfile with
`FROM ./base.<quant>.gguf` (no `ADAPTER` line — merged-only at this
upstream tag). Qwen2-VL's vision tower is dropped by the converter
and runs through Ollama's preprocessor path instead of an mmproj
sidecar, so `mmproj_path` is `null` in the export manifest; a future
tag that changes this would add a sidecar without breaking the
single-file contract. Emission is refused (with fallback to
HF-snapshot) when `--merged` is absent or `--imatrix` is not `off` —
the replay corpus is text-only and would mis-weight vision-adjacent
quant stats. **PARTIAL** (not yet seen for any registered base) would
mean the probe found only an `MmprojModel` registration for the arch.

Bump the vendored submodule (`scripts/bump-llama-cpp.sh bump <tag>`)
to refresh these verdicts; the bump script re-runs the probe and
rewrites the support JSON in the same commit.

## Refusal matrix

`dlm doctor` refuses VL training on:

- **CPU-only hosts.** PaliGemma fp16 inference on CPU takes minutes
  per generation step; training is impractical. No `--force` override.
- **CUDA hosts with < 12 GB VRAM.** Even LoRA batch=1 OOMs below that
  threshold.
- **MPS hosts with < 16 GB unified memory.** Same reasoning.

Override the last two with `--force` if you want to try anyway; the
first refusal stands.

## Preprocessing cache

The VL preprocessor (`dlm.data.vl_preprocessor`) caches its output
tensors under `~/.dlm/store/<dlm_id>/vl-cache/` keyed on
`(blob_sha, processor_sha, target_size)`. Per-image cache size scales
with the preprocessing plan:

| Base                      | Target size | Cache per image |
|---------------------------|------------:|----------------:|
| paligemma-3b-mix-224      |     224×224 |        ~0.5 MB  |
| internvl2-2b              |     448×448 |        ~2.0 MB  |
| qwen2-vl-2b-instruct      |     672×672 |        ~4.5 MB  |

A 100-image corpus on PaliGemma caches ~50 MB; the same corpus on
Qwen2-VL caches ~450 MB. Budget accordingly when running many
experiments.

Clear manually with `rm -rf ~/.dlm/store/<dlm_id>/vl-cache/` when
experimenting with different processors; the entries become stale
when `processor_sha` shifts (e.g. a transformers upgrade that
changes normalization constants).

## Related

- [Multi-modal training cookbook](../cookbook/multimodal-training.md)
- [Section format reference](../format/sections.md#image-image-path--alt--)
