# DocumentLanguageModel

> `.dlm` is a trainable local AI document format: typed sections, directives,
> replay-backed retraining, and export.

DocumentLanguageModel (DLM) is a local-first training, inference, and export
toolchain built around authored documents instead of hosted dashboards.

A `.dlm` can be:

- a hand-written training document with prose, instruction, and preference data
- a directive-driven entrypoint into a codebase or notes tree
- a multi-adapter project with learned routing
- a selected multimodal or audio-language document

DLM trains LoRA / QLoRA / DoRA adapters on real pretrained bases, keeps a replay
history so retrains do not silently forget, and exports local runtimes such as
Ollama and `llama-server`.

**Status:** pre-v1.0, but far beyond the original MVP framing. The core
author/train/prompt/export/pack/share loop is real, and newer runtime-target
work is landing incrementally. Current export targets are `ollama` and
`llama-server` (`llama-server` currently requires `--no-smoke` while the HTTP
smoke harness lands).

## What A `.dlm` Actually Is

A `.dlm` is not just “a text file with a special extension.”

It is a trainable project surface with:

- **frontmatter** for base-model choice, training config, export defaults,
  sources, cache policy, and multi-adapter gate settings
- **typed body sections** such as prose, `::instruction::`,
  `::preference::`, `::image::`, and `::audio::`
- **adapter routing** via fences like `::instruction#knowledge::`
- **directive-driven ingestion** from files and directories through
  `training.sources`
- **repo-local subtree control** through `.dlm/training.yaml` and `.dlm/ignore`
- a stable **`dlm_id`** that binds the document to a local store under
  `~/.dlm/store/<dlm_id>/`

That combination is what makes DLM more like a local AI authoring format than a
single prompt file.

## Why DLM

Most “personal AI” tooling still pushes you toward one of two bad choices:

- upload your data to someone else’s cloud
- run an oversized model with weak authoring and retraining ergonomics

DLM sits in the gap:

- **The document is the interface.** You author the thing you care about instead
  of wiring together a hidden dataset pipeline.
- **Training is real.** LoRA / QLoRA / DoRA on pretrained bases, not a toy
  from-scratch transformer.
- **Retraining is additive.** Previous document versions flow into a replay
  corpus so the model does not forget last week’s state by default.
- **Everything stays local.** Training, inference, store state, exports, and
  packs all live on your machine unless you explicitly push them somewhere.
- **Determinism is a contract.** Locks, pinned versions, and golden checks are
  first-class design constraints, not “best effort.”

## Core Capabilities

- **Author structured training data in one place.** Mix prose, SFT examples,
  preferences, image sections, and audio sections in one document.
- **Ingest whole trees, not just one file.** `training.sources` can walk a
  repo, and subtree-local `.dlm/training.yaml` / `.dlm/ignore` let the corpus
  carry its own curation rules.
- **Train on modern base families.** Text, reasoning-tuned, sparse-MoE,
  vision-language, and audio-language registry rows ship today, plus `hf:org/name`
  escape hatches.
- **Compose multiple adapters in one document.** Named adapters, weighted export
  mixes, and learned adapter gates let one `.dlm` separate knowledge, tone, or
  persona lanes.
- **Stay in a local iteration loop.** `dlm prompt`, `dlm repl`,
  `dlm train --watch`, `dlm metrics`, and `dlm doctor` are all part of the
  normal workflow now.
- **Export beyond the original Ollama-only story.** DLM still does explicit
  Ollama exports with pinned templates, and now also emits `llama-server`
  launch artifacts against the same GGUF path.
- **Close the eval loop.** `dlm harvest` can pull failing `sway`-style probe
  reports back into the document as new training examples.
- **Pack and share reproducibly.** `.dlm.pack`, verification, push/pull, and
  local serve flows are all built around the same store contracts.

## Supported Platforms

| Tier | Training | Inference / export |
|---|---|---|
| NVIDIA CUDA (SM ≥ 8.0) | bf16 + QLoRA 4-bit + FlashAttention | Ollama, GGUF export, `llama-server` launch artifacts |
| NVIDIA CUDA (SM < 8.0) | fp16 LoRA | Ollama, GGUF export, `llama-server` launch artifacts |
| Apple Silicon (MPS) | fp16 or fp32 LoRA depending on doctor plan | Ollama, selected MLX inference paths, GGUF export |
| CPU | inference-first; training refused above small bases unless forced | GGUF export, Ollama, `llama-server` launch artifacts |
| AMD ROCm | experimental | ROCm-oriented llama.cpp flows |

See [docs/hardware](./docs/hardware/memory-estimates.md) and
[docs/hardware/vl-memory.md](./docs/hardware/vl-memory.md) for the real support
matrix and current caveats.

## Install

### From the Homebrew tap

```sh
brew tap tenseleyFlow/tap
brew install dlm

# Optional, only if you want `--target ollama` registration/smoke:
brew install ollama
```

`brew install dlm` pulls in the Python environment and the vendored
`llama.cpp` source tree DLM uses for GGUF conversion. CUDA users unlock QLoRA
after install:

```sh
$(brew --prefix dlm)/libexec/venv/bin/pip install 'dlm[cuda]'
```

### From source

```sh
git clone https://github.com/tenseleyFlow/DocumentLanguageModel.git
cd DocumentLanguageModel
uv sync

# Build GGUF tooling:
scripts/bump-llama-cpp.sh build

# If you want the llama.cpp HTTP target too:
scripts/bump-llama-cpp.sh build --with-server

uv run dlm --help
```

We deliberately do not publish to PyPI yet. See
[CONTRIBUTING.md](./CONTRIBUTING.md) for the release flow.

## 30-Second Start

```sh
uv run dlm init tutor.dlm --base smollm2-135m
$EDITOR tutor.dlm
uv run dlm train tutor.dlm
uv run dlm prompt tutor.dlm "What is a Python decorator?"
uv run dlm export tutor.dlm --target ollama --name my-tutor
```

A minimal `.dlm` still works:

```dlm
---
dlm_id: 01KPM5CXB51GRX86Q25AKERN6E
dlm_version: 1
base_model: smollm2-135m
---

# Your document title

Write prose here.

::instruction::
### Q
What is a decorator?

### A
A function that takes a function and returns a wrapped function.
```

That path is still important. It is just no longer the whole story.

## Authoring Beyond The Toy Example

A more representative `.dlm` can mix directives, named adapters, and export
defaults in one place:

```dlm
---
dlm_id: 01KTESTEXAMPLE000000000000
dlm_version: 1
base_model: qwen3-1.7b
system_prompt: |
  You are a concise engineering assistant.
training:
  adapter: lora
  sequence_len: 4096
  sources_policy: strict
  sources:
    - path: ./src
      include: ["**/*.py", "**/*.md"]
      exclude: ["tests/**", "**/__pycache__/**"]
  adapters:
    knowledge:
      adapter: lora
      lora_r: 8
    tone:
      adapter: lora
      lora_r: 4
  gate:
    enabled: true
export:
  default_quant: Q4_K_M
---

# Project notes

Shared prose trains all declared adapters by default.

::instruction#knowledge::
### Q
What does the cache layer do?

### A
It avoids re-tokenizing unchanged directive-sourced files.

::preference#tone::
### Prompt
Explain a failure mode.

### Chosen
Explain it directly, then give the fix.

### Rejected
Over-explain the background before naming the problem.
```

Two important upgrades over the older README story:

- `training.sources` can turn a repo or notes tree into synthetic training
  sections.
- `training.adapters` + `training.gate` let one document route prompts across
  multiple adapters instead of pretending one flat adapter is the only mode.

If you need deeper subtree-specific curation, drop `.dlm/training.yaml` and
`.dlm/ignore` into nested directories and let the corpus carry its own rules.

## Common Workflows

### 1. Hand-authored document

```sh
uv run dlm init tutor.dlm --base smollm2-135m
uv run dlm train tutor.dlm
uv run dlm prompt tutor.dlm "Explain decorators"
```

### 2. Train across a codebase

```sh
uv run dlm train ./my-repo --base qwen3-1.7b --include '**/*.py' --name corpus
```

That auto-scaffolds a `.dlm` under `./my-repo/.dlm/` and lets the repo become
its own training surface.

### 3. Multi-adapter composition

```sh
uv run dlm prompt mydoc.dlm "Explain the runbook" --adapter knowledge
uv run dlm export mydoc.dlm --adapter-mix knowledge:1.0,tone:0.5
```

### 4. Local iteration loop

```sh
uv run dlm train mydoc.dlm --watch
uv run dlm repl mydoc.dlm
uv run dlm metrics mydoc.dlm
```

### 5. Export and ship

```sh
uv run dlm export mydoc.dlm --target ollama --name mydoc
uv run dlm export mydoc.dlm --target llama-server --no-smoke
uv run dlm pack mydoc.dlm --include-exports
uv run dlm verify mydoc.dlm.pack
```

### 6. Pull eval failures back into training

```sh
uv run dlm harvest mydoc.dlm --sway-json sway-report.json --apply
```

That is the probe-driven loop: evaluation finds a miss, DLM turns it into
document-level training data, and the next train closes the gap.

## Command Surface

The CLI is broader than the original MVP now. A useful mental map:

| Area | Commands | What they cover |
|---|---|---|
| Author | `init`, `templates`, `show`, `migrate`, `cache` | Create docs, inspect them, migrate schema, manage cache state |
| Train | `train`, `doctor`, `metrics`, `harvest` | Run training, inspect plans, observe runs, pull eval misses back in |
| Infer | `prompt`, `repl` | Local interactive and one-shot inference |
| Ship | `export`, `pack`, `unpack`, `verify`, `push`, `pull`, `serve` | Export to runtimes, bundle, verify, and move artifacts |

See the [CLI reference](./docs/cli/reference.md) for the full flag surface.

## Documentation

- [Getting started](./docs/getting-started/install.md)
- [Frontmatter reference](./docs/format/frontmatter.md)
- [Section grammar](./docs/format/sections.md)
- [Training across codebases](./docs/cookbook/training-across-codebases.md)
- [Multi-adapter composition](./docs/cookbook/multi-adapter.md)
- [Learned adapter gate](./docs/cookbook/learned-adapter-gate.md)
- [Multimodal training](./docs/cookbook/multimodal-training.md)
- [Audio training](./docs/cookbook/audio-training.md)
- [Probe-driven training / sway harvest](./docs/cookbook/probe-driven-training.md)
- [CLI reference](./docs/cli/reference.md)
- [Architecture](./docs/architecture.md)
- [Determinism](./docs/determinism.md)

## Principles

1. **The document is the interface.**
   But the document is structured: frontmatter, typed sections, directives, and
   store contracts all matter.
2. **Training is real.**
   LoRA / QLoRA / DoRA on pretrained bases, not a toy transformer.
3. **Retraining should not silently forget.**
   Replay-backed accumulation is part of the product.
4. **Local-first is load-bearing.**
   Your training data, adapters, exports, and packs stay on your machine unless
   you explicitly move them.
5. **Determinism is a contract.**
   If a change breaks the reproducibility story, that is a product regression.

## Tech Stack

Python 3.11+ · PyTorch · HuggingFace `transformers` / `peft` / `trl` /
`accelerate` / `datasets` · `watchfiles` · `prompt-toolkit` · `safetensors` ·
vendored `llama.cpp` for GGUF export · Ollama (optional runtime target) ·
Typer · Pydantic · `uv`

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). Testing conventions live in
[docs-internal/README-testing.md](./docs-internal/README-testing.md).

```sh
uv run pre-commit install
```

## License

MIT. Base-model licenses are separate and enforced where DLM needs them:
`dlm init`, `dlm train`, `dlm export`, and `dlm pack` all keep the gated-base
acceptance path explicit.
