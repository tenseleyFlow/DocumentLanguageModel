# DocumentLanguageModel

> A text file becomes your personal, locally-trained LLM.

Edit a `.dlm` file, train a LoRA on it, export to Ollama — all on your machine.
No telemetry, no uploads, no cloud. Built on PyTorch + HuggingFace with a
hardware-aware planner that picks precision, attention, and batching for your
box.

**Status:** pre-alpha. The foundation (CLI, document parser, content-addressed
store, hardware doctor) is landing now; real training lands next.

## What it does

- **Edit a document, get a model.** A `.dlm` is plain UTF-8 text with a YAML
  frontmatter and section fences (`::instruction::`, `::preference::`,
  default-prose). Prose trains via continued pretraining; instruction blocks
  train via SFT; preference blocks via DPO/ORPO (coming).
- **LoRA / QLoRA on a real base.** Curated registry of small pretrained bases
  (Qwen 2.5 0.5B–3B, Llama-3.2 1B/3B, SmolLM2 135M–1.7B, Phi-3.5-mini). Any
  HuggingFace model via an `hf:org/name` escape hatch.
- **Retrain, don't forget.** Prior document versions are stored in a
  zstd-compressed replay corpus and sampled back into each training run;
  retrains are additive, not destructive.
- **Deterministic by default.** Same document + same hardware tier + pinned
  versions → bit-identical adapter.
- **Export to Ollama.** `dlm export` produces a base GGUF + adapter GGUF +
  Modelfile with an explicit Go `text/template` (no fuzzy matching), then
  registers it locally with `ollama create`.
- **Hardware-aware.** `dlm doctor` picks precision (bf16 on Ampere+, fp16 on
  MPS), attention (FlashAttention when available, SDPA otherwise), batching,
  and gradient checkpointing.

## Supported platforms

| Tier | Training | Inference |
|---|---|---|
| NVIDIA CUDA (SM ≥ 8.0) | bf16 + QLoRA 4-bit + FlashAttention | Ollama (GGUF CUDA) |
| NVIDIA CUDA (SM < 8.0) | fp16 LoRA | Ollama (GGUF CUDA) |
| Apple Silicon (MPS) | fp16 LoRA | Ollama (GGUF Metal) |
| CPU | inference-only by default (training refused above 200M params) | Ollama (GGUF CPU) |
| AMD ROCm | experimental (later) | llama.cpp ROCm |

## Installation

```sh
# Requires Python 3.11+ and uv (https://github.com/astral-sh/uv)
git clone https://github.com/tenseleyFlow/DocumentLanguageModel.git
cd DocumentLanguageModel
uv sync
uv run dlm --help
```

For export: install [Ollama](https://ollama.com/) separately (minimum version
is pinned in the CLI; `dlm doctor` reports it).

## Quickstart

Once training lands (Sprint 9 — not shipped yet), the loop is:

```sh
uv run dlm init mydoc.dlm                 # scaffold a new .dlm
# edit mydoc.dlm — write prose, add ### Q / ### A pairs, etc.
uv run dlm train mydoc.dlm                # train a LoRA
uv run dlm prompt mydoc.dlm "question?"   # query the trained adapter
uv run dlm export mydoc.dlm --name mydoc  # register with Ollama
ollama run mydoc                          # use it
```

Today, `dlm doctor` and the `.dlm` parser surface are functional; other
subcommands are stubs that report which release will implement them.

## Principles

1. **The document is the interface.** Not a config file. Not a framework.
   Plain text with a special extension.
2. **Training is real.** LoRA/QLoRA on a pretrained base, not a toy
   from-scratch transformer.
3. **Retrain is additive.** Replay prior versions; never forget silently.
4. **Local-first, always.** Training, inference, and store all live on your
   disk. No network calls outside of model download.
5. **Deterministic by default.** Reproducibility is a contract, not a wish.

## Tech stack

Python 3.11+ · PyTorch · HuggingFace `transformers`/`peft`/`trl`/`accelerate` ·
bitsandbytes (CUDA-gated) · llama.cpp (vendored, for GGUF export) · Typer ·
Pydantic · `uv`.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). Testing conventions live at
[docs-internal/README-testing.md](./docs-internal/README-testing.md).

## License

MIT. Base-model licenses are separate and enforced at `dlm init` / `dlm pack`
time; Llama family bases require explicit acceptance.
