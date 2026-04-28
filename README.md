# DocumentLanguageModel

> A `.dlm` file becomes a local, reproducible, trainable LLM.
> Edit the document, retrain, share.

DocumentLanguageModel (DLM) is a local-first training, inference, and export
toolchain built around authored documents instead of hosted dashboards.

A `.dlm` can be:

- a hand-written training document with prose, instruction, and preference data
- a directive-driven entrypoint into a codebase or notes tree
- a multi-adapter project with learned routing
- a multimodal or audio-language document

DLM trains LoRA / QLoRA / DoRA adapters on real pretrained bases, keeps a replay
history so retrains do not silently forget, and exports to Ollama,
`llama-server`, `vllm`, and `mlx-serve`.

## Install

```sh
pip install document-language-model
```

That gives you the `dlm` command. Verify:

```sh
dlm --version
dlm doctor
```

### Extras

```sh
# CUDA QLoRA support (NVIDIA SM >= 8.0):
pip install 'document-language-model[cuda]'

# Apple Silicon MLX inference:
pip install 'document-language-model[mlx]'

# OpenAI teacher for synthetic data generation:
pip install 'document-language-model[openai]'

# Anthropic teacher:
pip install 'document-language-model[anthropic]'

# Observability (TensorBoard + W&B):
pip install 'document-language-model[observability]'
```

### From source

```sh
git clone https://github.com/tenseleyFlow/DocumentLanguageModel.git
cd DocumentLanguageModel
uv sync --all-extras --dev
uv run dlm --help

# Build GGUF export tooling:
scripts/bump-llama-cpp.sh build

# Optional: llama-server HTTP target:
scripts/bump-llama-cpp.sh build --with-server
```

## 30-Second Start

```sh
dlm init tutor.dlm --base smollm2-135m
# Edit tutor.dlm — add your Q&A pairs
dlm train tutor.dlm
dlm prompt tutor.dlm "What is a Python decorator?"
dlm export tutor.dlm --target ollama --name my-tutor
```

## What a `.dlm` Looks Like

A minimal document:

```yaml
---
dlm_id: 01KPM5CXB51GRX86Q25AKERN6E
dlm_version: 15
base_model: smollm2-135m
---

# My tutor

Some background prose. This trains via continued pretraining.

::instruction::
### Q
What is a decorator?

### A
A function that takes a function and returns a wrapped function.
```

A more representative one with directives, named adapters, and export config:

```yaml
---
dlm_id: 01KTESTEXAMPLE000000000000
dlm_version: 15
base_model: qwen3-1.7b
system_prompt: |
  You are a concise engineering assistant.
training:
  adapter: lora
  sequence_len: 4096
  sources:
    - path: ./src
      include: ["**/*.py", "**/*.md"]
      exclude: ["tests/**"]
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

## Common Workflows

### Train a hand-authored document

```sh
dlm init tutor.dlm --base smollm2-135m
dlm train tutor.dlm
dlm prompt tutor.dlm "Explain decorators"
```

### Train across a codebase

```sh
dlm train ./my-repo --base qwen3-1.7b
```

Auto-scaffolds a `.dlm` under `./my-repo/.dlm/` and trains on the repo's
source files.

### Multi-adapter composition

```sh
dlm prompt mydoc.dlm "Explain the runbook" --adapter knowledge
dlm export mydoc.dlm --adapter-mix knowledge:1.0,tone:0.5
```

### Export to local runtimes

```sh
dlm export mydoc.dlm --target ollama --name mydoc
dlm export mydoc.dlm --target llama-server
dlm export mydoc.dlm --target vllm
dlm export mydoc.dlm --target mlx-serve

# Also emit a ready-to-run sway.yaml next to the GGUF for downstream
# evaluation via `sway run` (requires the [sway] extra).
dlm export mydoc.dlm --target ollama --emit-sway-json
sway run <export-dir>/sway.yaml
```

### Mine preference pairs and retrain

```sh
dlm preference mine mydoc.dlm --samples 4 --max-pairs 8
dlm preference apply mydoc.dlm
dlm train mydoc.dlm --phase preference
```

### Generate synthetic training data

```sh
dlm synth instructions mydoc.dlm --teacher self --apply
dlm synth instructions mydoc.dlm --teacher openai:gpt-4o-mini --apply
```

### Multimodal and audio documents

```sh
dlm init diagrams.dlm --multimodal --base qwen2-vl-2b-instruct
dlm train diagrams.dlm
dlm prompt diagrams.dlm --image figures/arch.png "What is this?"

dlm init calls.dlm --audio
dlm train calls.dlm
dlm prompt calls.dlm --audio clips/call.wav "Summarize the clip"
```

### Pull eval failures back into training

```sh
dlm harvest mydoc.dlm --sway-json sway-report.json --apply
```

### Pack and share

```sh
dlm pack mydoc.dlm --include-exports
dlm verify mydoc.dlm.pack
dlm push mydoc.dlm --to hf:org/name
```

### Inspect state

```sh
dlm doctor
dlm show mydoc.dlm --json
dlm metrics mydoc.dlm
```

## Supported Platforms

| Tier | Training | Inference / Export |
|---|---|---|
| NVIDIA CUDA (SM >= 8.0) | bf16 + QLoRA 4-bit + FlashAttention | Ollama, GGUF, llama-server, vLLM |
| NVIDIA CUDA (SM < 8.0) | fp16 LoRA | Ollama, GGUF, llama-server, vLLM |
| Apple Silicon (MPS) | fp16 LoRA | Ollama, GGUF, MLX inference, mlx-serve |
| CPU | inference only (training refused above small bases) | GGUF, Ollama, llama-server |
| AMD ROCm | experimental | ROCm llama.cpp |

## Base Model Registry

DLM ships with ~27 pinned base models across text, vision-language, and
audio-language families:

- **Text:** Qwen 2.5 (0.5B–3B), Qwen 3 (1.7B–8B), Llama 3.2/3.3,
  SmolLM 2/3, Phi-3.5/4, Gemma 2, OLMo 2, Mixtral 8x7B
- **Vision-language:** Qwen2-VL, InternVL2/3, PaliGemma, Mistral-Small-3.1
- **Audio-language:** Qwen2-Audio

Any HuggingFace model via `--base hf:org/name` with compatibility probes.

## Command Surface

| Area | Commands |
|---|---|
| Author | `init`, `templates`, `show`, `migrate`, `cache` |
| Train | `train`, `doctor`, `metrics`, `harvest` |
| Align | `preference mine/apply/revert/list` |
| Synth | `synth instructions/preferences/revert/list` |
| Infer | `prompt`, `repl` |
| Ship | `export`, `pack`, `unpack`, `verify`, `push`, `pull`, `serve` |

See the [CLI reference](./docs/cli/reference.md) for the full flag surface.

## Editor support

### VSCode

Install **DLM — Document Language Model** from the
[VSCode Marketplace](https://marketplace.visualstudio.com/items?itemName=tenseleyFlow.dlm-vsc).
The extension provides syntax highlighting, completions, diagnostics, and a
side panel for `.dlm` authoring. Source:
[dlm-vsc](https://github.com/tenseleyFlow/dlm-vsc).

It uses the [dlm-lsp](https://github.com/tenseleyFlow/dlm-lsp) language
server, which you also need to install:

```sh
pip install dlm-lsp
```

### Other editors

The language server is editor-agnostic — Zed, Helix, and Neovim get
diagnostics, hover, and completions through their LSP clients. See:

- [Zed setup](./docs/cookbook/lsp-zed.md)
- [Helix setup](./docs/cookbook/lsp-helix.md)
- [Neovim setup](./docs/cookbook/lsp-neovim.md)

## Documentation

- [Getting started](./docs/getting-started/install.md)
- [Frontmatter reference](./docs/format/frontmatter.md)
- [Section grammar](./docs/format/sections.md)
- [CLI reference](./docs/cli/reference.md)
- [Training across codebases](./docs/cookbook/training-across-codebases.md)
- [Multi-adapter composition](./docs/cookbook/multi-adapter.md)
- [Multi-target export](./docs/cookbook/multi-target-export.md)
- [Self-improving loop](./docs/cookbook/self-improving-loop.md)
- [Synthesize training data](./docs/cookbook/synthesize-training-data.md)
- [Multimodal training](./docs/cookbook/multimodal-training.md)
- [Audio training](./docs/cookbook/audio-training.md)
- [Architecture](./docs/architecture.md)
- [Determinism](./docs/determinism.md)

## Principles

1. **The document is the interface.** Frontmatter, typed sections, directives,
   and store contracts — not a dashboard.
2. **Training is real.** LoRA / QLoRA / DoRA on pretrained bases.
3. **Retraining should not silently forget.** Replay-backed accumulation.
4. **Local-first is load-bearing.** Your data stays on your machine.
5. **Determinism is a contract.** Locks, pinned versions, golden checks.

## Tech Stack

Python 3.11+ · PyTorch · HuggingFace transformers / peft / trl / accelerate ·
vendored llama.cpp for GGUF · Ollama · Typer · Pydantic · uv

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT. Base-model licenses are separate and enforced at `dlm init`, `dlm train`,
`dlm export`, and `dlm pack`.
