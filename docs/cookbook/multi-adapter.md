# Multi-adapter composition

Train a single `.dlm` with more than one named adapter — keep knowledge
and tone orthogonal, mix them at export time, or prompt against one at
a time. Reach for this when you want separate "what the model knows"
and "how it says things" training signals without spinning up two
documents.

## When to use it

- A handbook where the **knowledge** is stable but the **tone** evolves
  (you might rewrite just the style examples next month).
- A single base model that needs to serve two personas — one customer-
  facing, one internal-engineering — where the instruction sets
  diverge.
- Experiments where you want to A/B two training recipes against the
  same prose corpus without forking the document.

If the answer is "one adapter is fine," skip this. Multi-adapter trades
simplicity for composition flexibility — pay that cost when you need
it.

## Document shape

```dlm
---
dlm_id: 01KPM618S7NXSPAY10BHKVECYX
base_model: qwen2.5-1.5b
training:
  sequence_len: 2048
  num_epochs: 2
  adapters:
    knowledge:
      adapter: lora
      lora_r: 8
    tone:
      adapter: lora
      lora_r: 4
      target_modules: [q_proj, v_proj]
      learning_rate: 1e-4
export:
  default_quant: Q4_K_M
---

# Domain prose

This prose trains BOTH adapters by default — prose without a `#name`
suffix fans out to every declared adapter. Most documents keep prose
shared so both adapters pick up the same domain vocabulary.

::instruction#knowledge::
### Q
What is the capital of France?
### A
Paris.

::instruction#tone::
### Q
How should I phrase things?
### A
Crisply. One sentence.
```

### Routing rules

| Section | Fence | Trains |
|---|---|---|
| Prose (no suffix) | `# heading` / plain prose | all adapters |
| Prose (pinned) | `::prose#knowledge::` | only `knowledge` |
| Instruction (no suffix) | `::instruction::` | first-declared adapter |
| Instruction (pinned) | `::instruction#tone::` | only `tone` |
| Preference | same — `::preference#name::` | only `name` |

The first-declared adapter acts as the implicit "default" for untagged
non-prose sections. Declaration order is the order you write them in
the YAML block.

## Training

One `dlm train` invocation trains all declared adapters:

```sh
$ uv run dlm train mydoc.dlm
```

Each adapter gets its own version history under
`~/.dlm/store/<dlm_id>/adapter/<name>/versions/vNNNN/` with an
independent `current.txt` pointer. The manifest grows one
`TrainingRunSummary` per adapter per invocation — running `dlm train`
again commits fresh `v0002` directories for each, never mixing lanes.

Each adapter is trained as a fresh LoRA from the base on its routed
rows; the base model loads once per adapter. Shared hyperparameters
(`sequence_len`, `num_epochs`, `seed`, optimizer, scheduler, warmup)
live at the `training` top level — per-adapter overrides are
intentionally limited to the LoRA-specific knobs.

## Prompting a specific adapter

```sh
$ uv run dlm prompt mydoc.dlm "Explain the runbook" --adapter knowledge
$ uv run dlm prompt mydoc.dlm "Explain the runbook" --adapter tone
```

`--adapter` is required on multi-adapter documents and rejected on
single-adapter ones. Unknown names get a clear error listing the
declared adapters.

## Exporting a specific adapter

```sh
$ uv run dlm export mydoc.dlm --adapter knowledge
```

One adapter → one Ollama model. The GGUF bundle + Modelfile embeds
that adapter only; `manifest.exports[-1].adapter_name` records which
one.

## Weighted composition at export

To ship a single Ollama model that combines both adapters:

```sh
$ uv run dlm export mydoc.dlm --adapter-mix knowledge:1.0,tone:0.5
```

This uses PEFT's `add_weighted_adapter` with linear combination to
produce a composite adapter, which is then converted to GGUF and
registered with Ollama as one unit.

Caveats (from the PEFT reference):

- **LoRA-only.** `add_weighted_adapter` doesn't support prefix / prompt
  tuning, and it can't merge across different LoRA ranks robustly. Keep
  all adapters in the mix on the same `adapter: lora` shape.
- **QLoRA requires dequantize.** Combining 4-bit quantized adapters
  into a composite is precision-unsafe; `dlm` refuses unless you pass
  `--dequantize` and `--merged` explicitly.
- **Mix is frozen in the export.** Once the Ollama model is built, the
  weights are baked. To change the mix, re-run `dlm export` with a new
  `--adapter-mix`. Ollama doesn't support hot-swapping adapter weights
  at runtime — keep the separate per-adapter exports around if you
  need dynamic composition at inference time.

## Hardware notes

`dlm doctor` refuses multi-adapter + QLoRA plans whose estimated VRAM
exceeds the device's 85% headroom (roughly: `base_4bit + 1 GB/adapter
+ 25% activations`). The failure points at two fixes: drop to
`adapter: lora` across the board, or reduce the adapter count. LoRA
multi-adapter plans are always accepted — each adapter's extra state
is negligible next to the base weights.

## When to fold back to a single adapter

Multi-adapter adds cognitive load and per-adapter training cost. Fold
back when:

- The adapters converge on similar behavior despite separate routing —
  the extra structure isn't doing work.
- One adapter's training set is so small (<10 rows) that it's adding
  noise instead of signal.
- Your export pipeline is always `--adapter-mix name:1.0,other:1.0` —
  a single adapter trained on the union is equivalent and cheaper.

## See also

- [Preference tuning (DPO vs ORPO)](preference-dpo-vs-orpo.md) —
  applies per-adapter on multi-adapter docs via `::preference#name::`
  routing.
- [Domain knowledge base](domain-kb.md) — the single-adapter story.
