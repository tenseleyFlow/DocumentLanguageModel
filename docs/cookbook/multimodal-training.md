# Multi-modal training (images + PaliGemma)

Sprint 35 v1 adds image sections to `.dlm` files. This recipe walks a
paper-figure corpus end-to-end: scaffold → drop images → train →
query the adapter against new images.

## Prerequisites

- Apple Silicon with ≥ 16 GB unified memory, or CUDA ≥ SM 8.0 with ≥
  12 GB VRAM. PaliGemma-3B-mix-224 fp16 fits inside both.
- A [Hugging Face account with the Gemma license
  accepted](https://huggingface.co/google/paligemma-3b-mix-224) and
  `HF_TOKEN` exported.
- PaliGemma cached locally (`huggingface-cli download
  google/paligemma-3b-mix-224`). First train attempt without this
  triggers the download automatically.

## Step 1 — Scaffold a VL `.dlm`

```bash
dlm init my-diagrams.dlm --multimodal --i-accept-license
```

`--multimodal` pins the base to `paligemma-3b-mix-224` and emits a
schema-v10 scaffold with a sample `::image::` fence. The initial
body references `figures/your-image.png` (non-existent by default —
drop real images into that path before the first train).

### Picking a different VL base

Three VL bases ship in the registry as of Sprint 35.3:

```bash
# Permissive + Apache-2.0 + strong general-purpose VL (pinned 672²):
dlm init my-diagrams.dlm --multimodal --base qwen2-vl-2b-instruct

# MIT-licensed, smallest per-image footprint (448²):
dlm init my-diagrams.dlm --multimodal --base internvl2-2b

# Default — Gemma license gate, cleanest PEFT path (224²):
dlm init my-diagrams.dlm --multimodal --i-accept-license
```

See [docs/hardware/vl-memory.md](../hardware/vl-memory.md) for the
VRAM table (inference / LoRA bs=1 / LoRA bs=4 per base) and the
base-selection matrix. InternVL2 has a loader caveat documented
there — it may need a transformers upgrade to load on older
installs.

## Step 2 — Author image sections

Two ways to add images. Either write them by hand:

```dlm
::image path="figures/architecture.png" alt="pipeline diagram"::
The retrieval pipeline: query → encoder → top-k → reranker → LLM.

::instruction::
### Q
What does this diagram show?

### A
A three-stage retrieval pipeline with reranking before the LLM.
```

Or ingest a directory through a source directive:

```dlm
---
dlm_id: 01JZ...
dlm_version: 10
base_model: paligemma-3b-mix-224
training:
  sources:
    - path: ./paper-figures
      include: ["**/*.png", "**/*.jpg"]
---
```

Each discovered image becomes an `::image::` section with `alt` set
to the filename stem and the caption empty (you can add prose
sections that reference the figures separately).

## Step 3 — Train

```bash
dlm train my-diagrams.dlm
```

The trainer:

1. Loads PaliGemma via `AutoModelForImageTextToText` + a matching
   `AutoProcessor`.
2. Walks `training.sources` directives, copies each image byte stream
   into the content-addressed blob store at
   `~/.dlm/store/<dlm_id>/blobs/`.
3. Emits training rows shaped `{images: [PIL], text: "<image>\n<caption>"}`.
4. Runs TRL 1.2's `DataCollatorForVisionLanguageModeling` — the built-in
   VL collator handles image-token expansion, pixel_values, and labels
   on-the-fly.
5. Commits the adapter under `adapter/versions/v0001/` just like the
   text path.

**Wall-clock expectations.** A 5-image corpus + 3 epochs on an M2
Pro (16 GB) takes about 60 minutes at `micro_batch_size=1` +
`grad_accum=4`. CUDA A100 with bf16 + batch=4 completes in ~5
minutes.

## Step 4 — Prompt the trained adapter

```bash
dlm prompt my-diagrams.dlm --image figures/architecture.png \
  "What does this diagram show?"
```

`--image` is required for VL bases. Repeat the flag for multi-image
prompts; each occurrence expands to one `<image>` placeholder the
processor slots pixels into.

## Step 5 — Export

Vision-language bases take the HF-snapshot path (GGUF conversion for
VL archs is in flux upstream; Sprint 35.4 adds the GGUF gate when
`llama.cpp`'s converter stabilizes):

```bash
dlm export my-diagrams.dlm
```

Writes to `~/.dlm/store/<dlm_id>/exports/hf-snapshot/`:

```
hf-snapshot/
  adapter/                  # PEFT LoRA weights
  processor/                # AutoProcessor config + tokenizer files
  snapshot_manifest.json    # export_target=hf_snapshot + sha256s
  README.md                 # how to load the snapshot downstream
```

To ship the snapshot somewhere, tar + send. To load it on the other
side:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

base = AutoModelForImageTextToText.from_pretrained(
    "google/paligemma-3b-mix-224",
    revision="8d2f7bc9c15d71a00c14f9eb7e4c7b99c79e0a11",
)
model = PeftModel.from_pretrained(base, "./adapter")
processor = AutoProcessor.from_pretrained("./processor")
```

The base isn't bundled — recipients download it on first use. Gemma
is `redistributable=False`; we can't legally ship its weights.

## Troubleshooting

### "no adapter under adapter/current.txt"

First train hasn't run. `dlm train my-diagrams.dlm` commits the
adapter; subsequent prompt/export calls need at least one run.

### "image not found: figures/your-image.png"

The `--multimodal` scaffold points at a placeholder; drop a real
image at that path, or edit the `::image path="..."::` fence to
reference a file that exists.

### "base {} is vision-language; pass at least one --image PATH"

You ran `dlm prompt` on a VL `.dlm` without attaching an image. VL
bases always expect an image token — even a throwaway question about
text content needs an image to anchor the placeholder.

### MPS out-of-memory during training

PaliGemma + batch=1 fits on 16 GB but leaves little headroom for
background processes. Close your browser, VS Code, etc. For
persistent OOM, swap to CUDA or wait for Sprint 35.4's quantization
support.

## What's not yet in Sprint 35 v1

- **Other VL bases.** Qwen2-VL-2B-Instruct + InternVL2-2B landed in
  Sprint 35.3 — use `--base qwen2-vl-2b-instruct` or `--base
  internvl2-2b`. See the base-selection section above.
- **Audio.** Sprint 35.2 ships `::audio path="..." transcript="..."::`.
- **GGUF export.** Sprint 35.4 adds `llama.cpp` arch detection + the
  Ollama Modelfile emitter. Until then, HF-snapshot is the only
  export target for VL.
- **Multi-image in one section.** Each `::image::` fence carries one
  image; prompts can stack multiple `<image>` tokens by repeating
  `--image` on the CLI.
