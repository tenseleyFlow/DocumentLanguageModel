# DocumentLanguageModel

> A text file becomes your personal, locally-trained LLM.

Edit a `.dlm` file, train a LoRA adapter on it, export to Ollama — all
on your machine. No telemetry, no uploads, no cloud. Built on PyTorch
+ HuggingFace with a hardware-aware planner that picks precision,
attention, and batching for your box.

**Status:** v1.0 release candidate. All Phase 3 sprints are complete;
the CLI surface (`init`, `train`, `prompt`, `export`, `pack`, `unpack`,
`doctor`, `show`, `migrate`) is wired end-to-end. A PyPI dry-run on
`test.pypi.org` is the last box to tick before the `v1.0` tag.

## Why

Most "personal AI" tooling either wants your data in their cloud or
asks you to run a 70B model you can't afford. DLM sits in the gap:
plain-text input, real pretrained bases (SmolLM2 for iteration, Qwen
or Llama for production), deterministic retraining, Ollama export.

- **Edit a document, get a model.** A `.dlm` is plain UTF-8 with a
  YAML frontmatter and section fences (`::instruction::`,
  `::preference::`, default-prose). Prose trains via continued
  pretraining; instruction blocks train via SFT; preference blocks via
  DPO/ORPO (Phase 4).
- **LoRA / QLoRA on a real base.** Curated registry of SmolLM2 135M–1.7B,
  Qwen 2.5 0.5B–3B, Llama-3.2 1B/3B, Phi-3.5-mini. Any HuggingFace
  model via an `hf:org/name` escape hatch.
- **Retrain, don't forget.** Prior document versions stay in a
  zstd-compressed replay corpus and get sampled into each training
  run. Edits are additive by default.
- **Deterministic by contract.** Same doc + same hardware tier +
  pinned versions → bit-identical adapter. `dlm.lock` records the
  tuple; `--strict-lock` upgrades every warn to an error. See
  [the determinism guide](./docs/determinism.md).
- **Explicit Ollama export.** `dlm export` emits a base GGUF +
  adapter GGUF + Modelfile with a pinned Go `text/template` (no
  fuzzy matching), then registers it via `ollama create`.
- **Hardware-aware.** `dlm doctor` probes the GPU, picks precision
  (bf16 on Ampere+, fp16 on MPS), attention (FlashAttention when
  available, SDPA otherwise), batching, and gradient checkpointing.

## Supported platforms

| Tier | Training | Inference |
|---|---|---|
| NVIDIA CUDA (SM ≥ 8.0) | bf16 + QLoRA 4-bit + FlashAttention | Ollama (GGUF CUDA) |
| NVIDIA CUDA (SM < 8.0) | fp16 LoRA | Ollama (GGUF CUDA) |
| Apple Silicon (MPS) | fp16 LoRA | Ollama (GGUF Metal) |
| CPU | inference-only by default (training refused above 200M params) | Ollama (GGUF CPU) |
| AMD ROCm | experimental (Phase 5) | llama.cpp ROCm |

## Install

### From source (current)

```sh
# Python 3.11+ and uv (https://github.com/astral-sh/uv)
git clone https://github.com/tenseleyFlow/DocumentLanguageModel.git
cd DocumentLanguageModel
uv sync
uv run dlm --help
```

### From PyPI (v1.0 target)

```sh
# Portable install — torch, transformers, peft, trl, datasets included.
pip install dlm

# Add the CUDA extra for QLoRA 4-bit on Ampere+.
pip install "dlm[cuda]"
```

For export: install [Ollama](https://ollama.com/) separately — minimum
version is pinned in the CLI; `dlm doctor` reports it. For GGUF
conversion, the repo vendors `llama.cpp` as a submodule; one-time
build:

```sh
scripts/bump-llama-cpp.sh build
```

## First run

```sh
$ uv run dlm init tutor.dlm --base smollm2-135m
init: wrote tutor.dlm
```

The scaffold:

```dlm
---
dlm_id: 01KPM5CXB51GRX86Q25AKERN6E
dlm_version: 1
base_model: smollm2-135m
---

# Your document title

Write prose here. It will train via continued pretraining (CPT) loss.

::instruction::

### Q
Your example question.

### A
Your example answer.
```

Open `tutor.dlm` in your editor, replace the placeholder content with
real prose + Q/A pairs, then:

```sh
$ uv run dlm train tutor.dlm
trained: v0001 (20 steps, seed=42, determinism=best-effort)
adapter: ~/.dlm/store/01KPM5…/adapter/versions/v0001
log:     ~/.dlm/store/01KPM5…/logs/train-000001-…jsonl

$ uv run dlm prompt tutor.dlm "What is a Python decorator?"
A decorator is a function that takes another function…

$ uv run dlm show tutor.dlm
/tmp/dlm-readme-demo/tutor.dlm
  dlm_id:         01KPM5CXB51GRX86Q25AKERN6E
  base_model:     smollm2-135m (revision 12fd25f)
  store:          ~/.dlm/store/01KPM5CXB51GRX86Q25AKERN6E  (537 B)
  adapter:        v0001
  training runs:  1
  exports:        0

$ uv run dlm export tutor.dlm --name my-tutor --quant Q4_K_M
export: base.Q4_K_M.gguf (47 MiB)
export: adapter.gguf (3 MiB)
export: Modelfile written; ollama create my-tutor:latest
export: smoke: "hello" → "Hi! How can I help?"

$ ollama run my-tutor "When should I use functools.wraps?"
Always, inside decorators. …
```

The [cookbook](./docs/cookbook/coding-tutor.md) has the walkthrough
for five starter scenarios (coding tutor, domain KB, writing partner,
personal assistant, changelog).

## Commands

Every command has `--help` for the full flag surface. Global flags
(`--home`, `-v`, `-q`, `--version`) apply to all subcommands.

| Command | Purpose | Key flags |
|---|---|---|
| `dlm init <path>` | Scaffold a new `.dlm` + create the store + record license acceptance. | `--base`, `--force`, `--i-accept-license` |
| `dlm train <path>` | Train / retrain the adapter. Replay-weighted by default. | `--resume`, `--fresh`, `--seed`, `--max-steps`, `--strict-lock`, `--update-lock`, `--ignore-lock` |
| `dlm prompt <path>` | Inference via HF (bypasses Ollama). Great for `--temp 0` determinism checks. | `--temp`, `--top-p`, `--max-tokens`, `--verbose` |
| `dlm export <path>` | Convert to GGUF, emit Modelfile, register with Ollama, smoke-run. | `--quant`, `--merged`, `--dequantize`, `--skip-ollama`, `--no-smoke`, `--no-imatrix`, `--draft` |
| `dlm pack <path>` | Bundle a `.dlm` + store into a portable `.dlm.pack`. | `--out`, `--include-exports`, `--include-base`, `--include-logs`, `--i-am-the-licensee` |
| `dlm unpack <pack>` | Restore a `.dlm.pack` into the local store. | `--force`, `--out` |
| `dlm doctor` | Probe hardware, print the resolved training plan. | `--json` |
| `dlm show <path>` | Training history + exports + adapter state. | `--json` |
| `dlm migrate <path>` | Upgrade a `.dlm` frontmatter to the current schema version. | `--dry-run`, `--no-backup` |

See the [CLI reference](./docs/cli/reference.md) for every flag + the
exit-code policy.

### Typical workflows

**Iterate on one document.** Edit, train, prompt, repeat:

```sh
$EDITOR tutor.dlm
uv run dlm train tutor.dlm          # additive retrain
uv run dlm prompt tutor.dlm "…"     # smoke
```

**Ship to Ollama.** Export, quant-level choice documented in the
[cookbook](./docs/cookbook/quantization-tradeoffs.md):

```sh
uv run dlm export tutor.dlm --quant Q4_K_M --name my-tutor
ollama run my-tutor
```

**Archive or share.** One-file bundle:

```sh
uv run dlm pack tutor.dlm --out tutor.dlm.pack           # ~100 MB (minimal)
uv run dlm pack tutor.dlm --include-exports --out tutor-full.dlm.pack
# …elsewhere:
uv run dlm unpack tutor-full.dlm.pack
```

**Start fresh.** Discard optimizer state + replay corpus:

```sh
uv run dlm train tutor.dlm --fresh
```

**Audit reproducibility.** Fail on any lock drift:

```sh
uv run dlm train tutor.dlm --strict-lock
```

## Documentation

- [Getting started](./docs/getting-started/install.md) — install →
  first train → first prompt → first export
- [The `.dlm` format](./docs/format/frontmatter.md) — frontmatter
  reference + section grammar
- [CLI reference](./docs/cli/reference.md) — every command, every flag
- [Cookbook](./docs/cookbook/coding-tutor.md) — 6 end-to-end recipes
- [Architecture](./docs/architecture.md) — module map + storage layout
  + contract boundaries
- [Determinism](./docs/determinism.md) — the reproducibility contract,
  severity table, regen-golden flow
- [Troubleshooting](./docs/troubleshooting.md) — symptom → cause →
  fix, seeded from the pitfall inventory

## Principles

1. **The document is the interface.** Not a config file. Not a
   framework. Plain text with a special extension.
2. **Training is real.** LoRA / QLoRA on a pretrained base, not a toy
   from-scratch transformer.
3. **Retrain is additive.** Replay prior versions; never silently
   forget.
4. **Local-first, always.** Training, inference, and store all live
   on your disk. No network calls outside of model download.
5. **Deterministic by default.** Reproducibility is a contract, not a
   wish. `dlm.lock` records the version tuple; drift fails loud.

## Tech stack

Python 3.11+ · PyTorch · HuggingFace `transformers` / `peft` / `trl` /
`accelerate` / `datasets` · `safetensors` · `bitsandbytes` (CUDA
extra) · vendored `llama.cpp` for GGUF export · Ollama (user-installed) ·
Typer · Pydantic · `packaging` · `uv`.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). Testing conventions live at
[docs-internal/README-testing.md](./docs-internal/README-testing.md).
Install the pre-commit hooks to match CI:

```sh
uv run pre-commit install
```

## License

MIT. Base-model licenses are separate and enforced at `dlm init` /
`dlm pack` time; Llama-family bases require explicit acceptance (see
`--i-accept-license`).
