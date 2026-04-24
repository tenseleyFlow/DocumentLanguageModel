# Multi-target export

`dlm export` is no longer just an Ollama registration path. The same
trained store can now emit local runtime artifacts for four targets:

- `ollama` for managed local registration plus the existing Modelfile flow
- `llama-server` for GGUF-backed OpenAI-compatible HTTP serving via vendored
  `llama.cpp`
- `vllm` for HF-snapshot plus LoRA-module serving on machines that can run
  `vllm`
- `mlx-serve` for Apple Silicon text serving through `mlx_lm.server`

Use this when you want one training loop but different local runtimes for
prompting, evaluation harnesses, agents, or deployment experiments.

## Quick map

| Target | Best for | Artifact shape | Smoke path |
|---|---|---|---|
| `ollama` | Easiest local chat loop | GGUF + `Modelfile` + local registration | existing Ollama smoke |
| `llama-server` | GGUF-backed OpenAI-compatible server | `base.<quant>.gguf` + `adapter.gguf` + `chat-template.jinja` + `llama-server_launch.sh` | shared HTTP smoke |
| `vllm` | HF-snapshot + LoRA serving on supported hosts | `vllm_launch.sh` + `vllm_config.json` + staged adapters | shared HTTP smoke |
| `mlx-serve` | Apple Silicon text serving without GGUF conversion | `mlx_serve_launch.sh` + staged MLX adapter dir | shared HTTP smoke |

## Prerequisites

### Ollama

```sh
brew install ollama
```

### llama-server

```sh
scripts/bump-llama-cpp.sh build --with-server
```

That compiles the vendored `llama-server` binary alongside the GGUF tooling.

### vLLM

Install a compatible `vllm` runtime in the environment you plan to launch
from. DLM writes the launch/config artifacts, but it does not bundle the
server runtime.

On Apple Silicon, the generated `vllm` launch path is deliberately cautious:

- `VLLM_METAL_USE_PAGED_ATTENTION=0`
- `VLLM_METAL_MEMORY_FRACTION=auto`
- `--max-model-len` capped to the document's `training.sequence_len`

Those defaults exist to avoid the Metal OOM / hang pattern that shows up when
`vllm-metal` blindly asks for the base model's full context window.

### MLX-serve

```sh
uv sync --extra mlx
```

`mlx-serve` is Apple Silicon only. DLM refuses it on CUDA, ROCm, and CPU-only
hosts, and this Sprint 41 slice only supports text bases on that target.

## Common exports

### Ollama

```sh
uv run dlm export tutor.dlm --target ollama --name my-tutor
```

This is the classic DLM path: GGUF conversion, explicit Go-template
`Modelfile`, optional registration, and an Ollama smoke prompt.

### llama-server

```sh
uv run dlm export tutor.dlm --target llama-server
bash ~/.dlm/store/<dlm_id>/exports/Q4_K_M/llama-server_launch.sh
```

This reuses the GGUF export artifacts and adds:

- `chat-template.jinja`
- `llama-server_launch.sh`
- `target: "llama-server"` in `export_manifest.json`

The launch script binds `127.0.0.1` and speaks `/v1/chat/completions`.

### vLLM

```sh
uv run dlm export tutor.dlm --target vllm
bash ~/.dlm/store/<dlm_id>/exports/vllm/vllm_launch.sh
```

This path stages local LoRA modules and writes:

- `vllm_launch.sh`
- `vllm_config.json`
- `exports/vllm/adapters/...`

Flags that only matter to GGUF or Ollama are ignored with a banner:
`--quant`, `--merged`, `--dequantize`, `--no-template`, `--skip-ollama`,
`--no-imatrix`, `--draft`, `--no-draft`.

### MLX-serve

```sh
uv run dlm export tutor.dlm --target mlx-serve
bash ~/.dlm/store/<dlm_id>/exports/mlx-serve/mlx_serve_launch.sh
```

This path stages an MLX-loadable adapter directory and writes:

- `mlx_serve_launch.sh`
- `exports/mlx-serve/adapter/` or one named adapter directory
- `target: "mlx-serve"` in `export_manifest.json`

`mlx-serve` also ignores the GGUF/Ollama-only flags above, plus `--name`.

## Multi-adapter behavior

The runtime targets split into two families:

- `ollama` and `llama-server` can reuse the GGUF weighted-merge path for
  `--adapter-mix`
- `vllm` and `mlx-serve` work from local adapter directories

For `vllm`:

- single-adapter docs export one staged module
- multi-adapter docs without `--adapter` export every named adapter as a
  `--lora-modules` list
- `--adapter-mix` exports the staged composite adapter instead

For `mlx-serve`:

- single-adapter docs export the current flat adapter
- multi-adapter docs must choose one adapter with `--adapter`, or pass
  `--adapter-mix` to export the staged composite adapter

That "one adapter at a time" rule is intentional: this target is a simple
local-serving path, not a dynamic multi-LoRA router.

## Smoke behavior

All three HTTP targets use the shared OpenAI-compatible smoke harness:

1. reserve a loopback port
2. launch the target-specific server command
3. poll `/v1/models`
4. POST `/v1/chat/completions`
5. record the first non-empty line in the store manifest

Skip it with `--no-smoke` when the runtime is not installed or you want the
artifacts only.

## Inspecting what got written

Every export writes `export_manifest.json` under its target directory. The
important fields are:

- `target`
- `quant`
- `artifacts`
- `adapter_version`
- `base_model_hf_id`
- `base_model_revision`

The per-store `manifest.json` also gets an appended `exports[-1]` row with the
same `target` plus the smoke first line when a smoke test ran.

See [Export manifest](../format/export-manifest.md) for the exact schema.
