# First export

`dlm export` converts the base + adapter into GGUF files, writes a
Modelfile with an explicit Go `text/template` (no fuzzy matching),
registers the model with `ollama create`, and runs a smoke prompt.

## Prerequisites

- `vendor/llama.cpp` submodule is built:
  ```sh
  $ scripts/bump-llama-cpp.sh build
  ```
  This compiles `llama-quantize` and `llama-imatrix` under
  `vendor/llama.cpp/build/bin/`.

- [Ollama](https://ollama.com/) is installed and its daemon is running.
  `dlm doctor` reports the minimum version.

## Export

```sh
$ uv run dlm export tutor.dlm --quant Q4_K_M --name my-tutor
export: preflight ok
export: base.Q4_K_M.gguf (47 MiB)
export: adapter.gguf (3 MiB)
export: Modelfile written; ollama create my-tutor:latest
export: smoke: "Hi!" → "Hello! How can I help?"
manifest: exports[-1] recorded at ~/.dlm/store/01KC…/
```

Under the hood:

1. The export **preflight** (Sprint 11) checks the adapter config
   matches the base architecture, asserts the tokenizer vocab agrees
   with the base, validates the chat template, and confirms the
   adapter wasn't QLoRA-trained (pitfall #3 — QLoRA merge needs
   `--dequantize`).
2. The base model is converted to GGUF and quantized via
   `llama-quantize`. The GGUF is cached under
   `~/.dlm/store/<id>/exports/Q4_K_M/base.Q4_K_M.gguf` — subsequent
   exports at the same quant reuse the file.
3. The LoRA adapter is converted to `adapter.gguf`.
4. An explicit `Modelfile` is emitted with `FROM`, `ADAPTER`, and an
   explicit `TEMPLATE "..."` directive (Sprint 12). Ollama will **not**
   fuzzy-match the template — the exact Go template for the base's
   dialect is committed.
5. `ollama create <name>:latest` registers the model under the Ollama
   daemon's control.
6. A smoke prompt runs; the first line of output is recorded in
   `manifest.exports[-1].smoke_output_first_line`.

## Quant levels

| Quant | Size | Quality | When to use |
|---|---|---|---|
| `Q4_K_M` | ~50% of fp16 | Great default | General-purpose; recommended starting point. |
| `Q5_K_M` | ~60% | Higher quality | Willing to trade more disk for fidelity. |
| `Q8_0` | ~100% of int8 | Near-lossless | Baseline for quality comparisons. |
| `F16` | 100% | No quantization | Debugging a quant-caused regression. |

See [Quantization tradeoffs](../cookbook/quantization-tradeoffs.md) for
a deeper dive.

## imatrix-calibrated quantization

If your store has a replay corpus with enough signal (Sprint 11.6),
the export runner automatically builds an imatrix from it and passes
`--imatrix` to `llama-quantize`. This gives noticeable quality
improvements on `Q4_K_M` and below without changing the API.

Opt out with `--no-imatrix` if you'd rather have a static quant for
comparison.

## Just produce GGUFs, skip Ollama

```sh
$ uv run dlm export tutor.dlm --quant Q4_K_M --skip-ollama
```

Useful on CI runners without the Ollama daemon installed. The GGUFs
land in `exports/Q4_K_M/`; wire them into your own runtime.

## Next

Want to send the whole training history to a friend? The
[Sharing with pack](../cookbook/sharing-with-pack.md) cookbook shows
the `dlm pack` / `dlm unpack` round trip.
