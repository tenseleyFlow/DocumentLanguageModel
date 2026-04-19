# Install

DocumentLanguageModel is a Python package. It depends on `torch` (GPU or
CPU build), `transformers`, `peft`, `trl`, and — optionally for export —
the `ollama` binary on your PATH.

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.11 | `pyproject.toml` pins `python >= 3.11`. |
| [uv](https://github.com/astral-sh/uv) | any recent | Used for dependency resolution and running scripts. |
| PyTorch | 2.4+ | Installed automatically by `uv sync`. |
| Ollama | as reported by `dlm doctor` | Only needed for `dlm export` smoke runs. |
| `vendor/llama.cpp` submodule | built | Only needed for `dlm export`. `scripts/bump-llama-cpp.sh build` compiles `llama-quantize` + `llama-imatrix`. |

On Apple Silicon, MPS acceleration is detected automatically and DLM
plans for fp16 LoRA. On CUDA, compute capability ≥ 8.0 (Ampere and
newer) unlocks bf16 + QLoRA 4-bit. See [Architecture](../architecture.md)
for the full refusal matrix.

## Install from source

```sh
git clone https://github.com/tenseleyFlow/DocumentLanguageModel.git
cd DocumentLanguageModel
uv sync
uv run dlm --help
```

`uv sync` resolves the dependency tree into `.venv/` and pulls the
pinned versions from `uv.lock`. Use `uv run dlm <command>` (not
`dlm <command>` — the CLI isn't on your shell PATH unless you activate
the venv).

## Install from PyPI

```sh
# Coming with v1.0 — the tagged release workflow publishes to PyPI via
# trusted-publisher OIDC. Until then, install from source.
pip install dlm
```

## Verify

```sh
$ uv run dlm --version
dlm 0.1.0

$ uv run dlm doctor
backend: mps
precision: fp16
attn:     sdpa
...
```

`dlm doctor` is the first command to run on a new machine. It probes
the GPU, reports the memory budget, picks a training plan, and warns
about anything missing (e.g. FlashAttention unavailable, bitsandbytes
not importable on CPU-only hosts).

## Next

Got `dlm doctor` output that looks healthy? Move on to the
[first training cycle](first-train.md).
