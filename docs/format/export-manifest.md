# Export manifest

Every `dlm export` writes an `export_manifest.json` inside the export directory.
It is the target-local record of what DLM emitted, separate from the broader
per-store `manifest.json`.

Examples:

- `~/.dlm/store/<dlm_id>/exports/Q4_K_M/export_manifest.json`
- `~/.dlm/store/<dlm_id>/exports/vllm/export_manifest.json`
- `~/.dlm/store/<dlm_id>/exports/mlx-serve/export_manifest.json`

## What it records

The manifest captures:

- `target`: which runtime this export was prepared for
- `quant`: the export family (`Q4_K_M`, `Q8_0`, `hf`, ...)
- `merged` / `dequantized`: whether LoRA weights were merged into the base
- `created_at` and `created_by`
- `llama_cpp_tag` when the target depends on vendored `llama.cpp`
- `base_model_hf_id` and `base_model_revision`
- `adapter_version`
- `artifacts`: every emitted file with relative path, sha256, and size

The schema is strict and round-trips through the Pydantic model in
`src/dlm/export/manifest.py`.

## Example

```json
{
  "target": "llama-server",
  "quant": "Q4_K_M",
  "merged": false,
  "dequantized": false,
  "ollama_name": null,
  "created_at": "2026-04-23T18:42:00",
  "created_by": "dlm-0.1.0",
  "llama_cpp_tag": "b4281",
  "base_model_hf_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "base_model_revision": "4c0d2...",
  "adapter_version": 3,
  "artifacts": [
    {
      "path": "base.Q4_K_M.gguf",
      "sha256": "…",
      "size_bytes": 47211904
    },
    {
      "path": "adapter.gguf",
      "sha256": "…",
      "size_bytes": 3145728
    },
    {
      "path": "llama-server_launch.sh",
      "sha256": "…",
      "size_bytes": 312
    }
  ]
}
```

## `target`

`target` is now the load-bearing field for Sprint 41’s runtime split.

Current values:

- `ollama`
- `llama-server`
- `vllm`
- `mlx-serve`

That lets downstream tooling distinguish:

- a GGUF + Modelfile export meant for Ollama
- a GGUF-backed OpenAI-compatible launch artifact set
- an HF-snapshot + LoRA-module export for `vllm`
- an MLX adapter export for Apple Silicon serving

## Relationship to the store manifest

`export_manifest.json` is per-export and artifact-focused.

The store-level `manifest.json` keeps the running narrative in `exports[]`:

- when the export happened
- which `target` it used
- GGUF checksums when present
- `ollama_name` when relevant
- the first smoke output line when a smoke test ran

Use `export_manifest.json` when you need exact artifact provenance for one
export directory. Use `manifest.json` when you want the store’s full history.
