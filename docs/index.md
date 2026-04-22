# DocumentLanguageModel

> A text file becomes your personal, locally-trained LLM.

Edit a `.dlm` file, train a LoRA on it, export to Ollama — all on your
machine. No telemetry, no uploads, no cloud.

DocumentLanguageModel (DLM) is built on PyTorch + HuggingFace with a
hardware-aware planner that picks precision, attention, and batching
for your box. Retraining is additive: prior document versions stay in a
replay corpus so the model doesn't forget what you taught it last week.

## Why

Most "personal AI" tools either want your data in their cloud or ask
you to run a 70B model you can't afford. DLM sits in the gap:

- **Your document is the dataset.** The `.dlm` file under version
  control is both the prose you're training on and the configuration
  for how the training runs. Edit, retrain, share.
- **Real pretrained bases.** SmolLM2-135M for fast iteration; newer
  registry rows like Qwen3, Llama 3.3, Gemma 2, SmolLM3, Phi-4-mini-
  reasoning, OLMo-2, Mixtral, and Mistral Small 3.1 cover current
  text, sparse-MoE, and multimodal use cases. No from-scratch
  transformers, no toy experiments.
- **Deterministic by contract.** Same document + same hardware tier +
  pinned versions produce bit-identical adapters. [Determinism](determinism.md)
  is a first-class feature.
- **Exports to Ollama.** `dlm export` emits a quantized GGUF, an
  explicit Go-template `Modelfile` (no fuzzy matching), and registers
  the model locally. Share the adapter or pack the whole training
  history into one `.dlm.pack` file.

## 30-second demo

```sh
$ uv run dlm init tutor.dlm --base smollm2-135m
created: tutor.dlm

$ $EDITOR tutor.dlm     # write some Q&A under ::instruction::

$ uv run dlm train tutor.dlm
trained: v0001 (20 steps, seed=42, determinism=best-effort)

$ uv run dlm prompt tutor.dlm "Explain Python decorators"
A decorator is a function that takes a function and returns …

$ uv run dlm export tutor.dlm --name my-tutor
ollama: registered my-tutor:latest
```

## Where to next

| If you want to… | Start here |
|---|---|
| Install DLM and run the first cycle | [Getting started → Install](getting-started/install.md) |
| Understand the `.dlm` file format | [The `.dlm` format](format/frontmatter.md) |
| See every CLI command | [CLI reference](cli/reference.md) |
| Copy a working recipe | [Cookbook](cookbook/coding-tutor.md) |
| Debug a confusing failure | [Troubleshooting](troubleshooting.md) |

## Status

DLM is pre-v1.0 as of 2026-04-19. Phase 3 (MVP release) sprints are
complete through **Sprint 15 — Reproducibility lock**; Sprint 16
(this documentation, plus the release workflow) is in progress. See
[Architecture](architecture.md) for the full sprint map.
