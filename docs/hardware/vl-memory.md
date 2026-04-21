# Vision-language memory budget

Sprint 35 v1 ships one VL base — **PaliGemma-3B-mix-224**. Further VL
bases land in Sprint 35.3; their budgets land here as they're
validated.

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
`(blob_sha, processor_sha, target_size)`. Budget ~0.5 MB per 224×224
image after preprocessing. A 100-image corpus caches ~50 MB.

Clear manually with `rm -rf ~/.dlm/store/<dlm_id>/vl-cache/` when
experimenting with different processors; the entries become stale
when `processor_sha` shifts (e.g. a transformers upgrade that
changes normalization constants).

## Related

- [Multi-modal training cookbook](../cookbook/multimodal-training.md)
- [Section format reference](../format/sections.md#image-image-path--alt--)
