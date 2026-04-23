# Memory estimates

These are planning numbers, not a promise. DLM’s doctor still does the
real refusal/fit decision, but the table below is the quick mental map
for the Sprint 40 refresh rows that changed the most user expectations.

## Text-family checkpoints

| Base | fp16 weights | Practical target |
|---|---:|---|
| `qwen3-8b` | ~16 GB | 24 GB CUDA or high-memory Apple Silicon for LoRA; lighter inference on smaller boxes. |
| `llama-3.3-8b-instruct` | ~16.5 GB | Same class as other 8B text rows: real GPU planning required for training. |
| `gemma-2-9b-it` | ~18 GB | 24 GB CUDA is the comfortable floor. |
| `mistral-small-3.1-24b-instruct` | ~48 GB | Large-CUDA-first. Refused on MPS by default unless forced. |

## What the doctor is approximating

For LoRA/QLoRA, the planner estimates:

- base weights at the chosen load precision
- activation memory from `sequence_len × micro_batch × layers`
- optimizer state for the trainable adapter params
- LoRA parameter storage
- a 20% safety margin on top

That estimator lives in `src/dlm/hardware/memory.py` and is intentionally conservative.

## Rules of thumb

- 8B-class rows are where laptop experimentation starts turning into real hardware planning.
- 9B-class rows are usually fine on 24 GB CUDA, but not “casual” on smaller hosts.
- 24B-class rows are not broad consumer defaults. In DLM they are treated as explicit high-capacity picks.
- MPS can be surprisingly good for text LoRA, but DLM now refuses oversized bases like `mistral-small-3.1-24b-instruct` by default because unified memory headroom disappears too quickly.

## Related

- [Choosing a base](../cookbook/choosing-a-base.md)
- [Vision-language memory budget](vl-memory.md)
