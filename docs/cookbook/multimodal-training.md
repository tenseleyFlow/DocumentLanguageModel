# Multi-modal training (images + VL bases)

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

Five VL bases ship in the registry today:

```bash
# Permissive + Apache-2.0 + strong general-purpose VL (pinned 672²):
dlm init my-diagrams.dlm --multimodal --base qwen2-vl-2b-instruct

# MIT-licensed, smallest per-image footprint (448²):
dlm init my-diagrams.dlm --multimodal --base internvl2-2b

# Newer InternVL planning row (dynamic 448-tiling, still runtime-deferred):
dlm init my-diagrams.dlm --multimodal --base internvl3-2b

# Largest-capability VL row, CUDA-first (pinned 1540²):
dlm init my-diagrams.dlm --multimodal --base mistral-small-3.1-24b-instruct

# Default — Gemma license gate, cleanest PEFT path (224²):
dlm init my-diagrams.dlm --multimodal --i-accept-license
```

See [docs/hardware/vl-memory.md](../hardware/vl-memory.md) for the
VRAM table (inference / LoRA bs=1 / LoRA bs=4 per base) and the
base-selection matrix. **Heads-up on InternVL2**: the row is visible in
the registry, but on the current stack DLM now refuses it for actual
prompt/train/HF-snapshot-export work. The upstream family still needs a
custom processor/collator path for its tokenizer-only `AutoProcessor`,
`<image>` expansion, and `image_flags` forward contract. The same
family gap applies to `internvl3-2b` as well: it is now registry-
visible and scaffoldable, but the generic runtime still refuses the
whole InternVL family until DLM owns that custom contract.
**Heads-up on Mistral Small 3.1**: it is a real VL registry row now,
but it is intentionally treated as a large-CUDA-first base. `dlm
doctor` refuses it on Apple Silicon by default unless you explicitly
pass `--force` on a large-memory host.

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
   `AutoProcessor` (or the equivalent generic VL processor for Qwen2-VL
   / Mistral Small 3.1).
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

`dlm export` on a VL base probes the vendored llama.cpp for GGUF
coverage of the base's arch class and routes to one of three paths:

- **SUPPORTED** — llama.cpp's `convert_hf_to_gguf.py` registers the
  arch (the LM side converts cleanly). The export path will emit
  GGUF + an Ollama-compatible Modelfile once the single-file VL
  emission hook lands in dlm. Today the dispatcher falls through to
  HF-snapshot with a banner noting the status. Of the three
  registered VL bases, **qwen2-vl-2b-instruct** and
  **mistral-small-3.1-24b-instruct** are SUPPORTED at the current
  vendored tag.
- **PARTIAL** — the arch is registered only on an `MmprojModel`
  subclass; the vision tower converts but no single-file GGUF covers
  the full VL model. Falls back to HF-snapshot with a PARTIAL banner.
  None of the registered bases hit this verdict at the pinned tag.
- **UNSUPPORTED** — llama.cpp doesn't know the arch at all. Falls
  back to HF-snapshot with an actionable banner naming the arch
  class and the vendored tag. **paligemma-3b-mix-224**,
  **internvl2-2b**, and **internvl3-2b** are UNSUPPORTED at the
  pinned tag.

See [docs/hardware/vl-memory.md](../hardware/vl-memory.md#llamacpp-gguf-support-matrix-sprint-354)
for the current support verdicts; bump the vendored tag with
`scripts/bump-llama-cpp.sh bump <tag>` to refresh (the script re-runs
the arch probe + rewrites the support JSON in the same commit).

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
persistent OOM, swap to CUDA (VL QLoRA is a planned follow-up).

If you're trying `mistral-small-3.1-24b-instruct`, this is expected to
be much stricter: the current planner refuses that base on Apple
Silicon by default unless you pass `--force` on a large-memory host.

### "InternVL-family runtime still needs a custom collator path"

That refusal is deliberate. The current generic VL stack assumes a real
image processor + TRL's built-in vision collator. InternVL-family bases
still expose a tokenizer-only `AutoProcessor` on this stack and rely on
custom `<image>` expansion plus `image_flags`. The registry row stays
visible for planning and future work, but use the other VL bases for
actual runs today.

## Known limitations

- **Multi-image in one section.** Each `::image::` fence carries one
  image; prompts can stack multiple `<image>` tokens by repeating
  `--image` on the CLI.
- **Audio ingest.** Audio is a separate path —
  `::audio path="..." transcript="..."::` on an audio-language base.
  See [audio-training.md](audio-training.md).

## VL GGUF emitter trajectory

The VL export path today routes every verdict through HF-snapshot
and prints a banner. Going from that to single-file VL GGUF needs
three pieces to line up, in order:

1. **Upstream llama.cpp** registers the VL arch class in
   `convert_hf_to_gguf.py` (currently only Qwen2-VL; PaliGemma and
   InternVL2 are UNSUPPORTED at the pinned tag). Our
   `scripts/bump-llama-cpp.sh` re-runs the arch probe on every bump
   and caches verdicts in `vendor/llama_cpp_vl_arch_support.json`,
   so re-verdicting is mechanical once a new llama.cpp tag lands.
2. **The dlm-side emitter** invokes the upstream converter on a
   merged VL adapter, packages the resulting GGUF, and hands it to
   `render_vl_modelfile` for the Ollama-compatible Modelfile. The
   renderer, arch probe, version guard, and per-family stops are
   already in place; only the emitter orchestration is missing.
3. **An integration test** picks one SUPPORTED base, trains a
   1-step adapter on the fixture, converts to GGUF, runs
   `ollama create`, and smoke-tests inference. The test scaffold
   (auto-skip while UNSUPPORTED) is already checked in; the body
   fills in when step 2 lands.

Until all three align, `dlm export` on a VL base writes an
HF-snapshot tarball — the same artifact a downstream recipient loads
via `AutoModelForImageTextToText.from_pretrained` +
`PeftModel.from_pretrained`. See
[docs/hardware/vl-memory.md](../hardware/vl-memory.md#llamacpp-gguf-support-matrix-sprint-354)
for the current per-arch verdicts.
