# Frontmatter reference

The YAML block between the two `---` lines at the top of every `.dlm`
document. Validated with Pydantic in `dlm.doc.schema` (`extra="forbid"`,
`frozen=True`) — unknown keys or wrong types fail fast with a
`file:line:col` error.

## Minimum required frontmatter

```yaml
---
dlm_id: 01HRZYQ2X0MB5K4VN7E9DNT5GH
base_model: smollm2-135m
---
```

`dlm_id` is a 26-character Crockford base32 ULID. `dlm init` generates
it; don't edit it by hand.

`base_model` is either a registry key or `hf:org/name`:

| Registry key | HuggingFace id |
|---|---|
| `smollm2-135m` | HuggingFaceTB/SmolLM2-135M-Instruct |
| `smollm2-360m` | HuggingFaceTB/SmolLM2-360M-Instruct |
| `smollm2-1.7b` | HuggingFaceTB/SmolLM2-1.7B-Instruct |
| `qwen2.5-0.5b` | Qwen/Qwen2.5-0.5B-Instruct |
| `qwen2.5-1.5b` | Qwen/Qwen2.5-1.5B-Instruct |
| `qwen2.5-3b` | Qwen/Qwen2.5-3B-Instruct |
| `qwen2.5-coder-1.5b` | Qwen/Qwen2.5-Coder-1.5B-Instruct |
| `llama-3.2-1b` | meta-llama/Llama-3.2-1B-Instruct (gated) |
| `llama-3.2-3b` | meta-llama/Llama-3.2-3B-Instruct (gated) |
| `phi-3.5-mini` | microsoft/Phi-3.5-mini-instruct |

Off-registry bases use `hf:` prefix, e.g.
`base_model: hf:mistralai/Mistral-7B-Instruct-v0.3`. `dlm init` runs
a compatibility probe; failures abort with a clear diagnostic.

## Full frontmatter

```yaml
---
dlm_id: 01HRZYQ2X0MB5K4VN7E9DNT5GH
dlm_version: 1                    # bumped by `dlm migrate`; default: 1
base_model: qwen2.5-1.5b
system_prompt: |
  You are a concise assistant.
training:
  adapter: lora                   # or qlora (CUDA only)
  lora_r: 8                       # 1..256
  lora_alpha: 16
  lora_dropout: 0.05              # 0.0..0.5
  target_modules: auto            # or a list[str]
  sequence_len: 2048              # 64..32768
  micro_batch_size: auto          # or a positive int
  grad_accum: auto                # or a positive int
  learning_rate: 2e-4
  num_epochs: 3
  optimizer: adamw_torch          # or adamw_bnb_8bit / paged_adamw_8bit
  lr_scheduler: cosine            # or linear / constant
  warmup_ratio: 0.1               # 0.0..0.5
  # precision: fp16               # optional override; default lets the doctor pick
  seed: 42
export:
  default_quant: Q4_K_M           # or Q5_K_M / Q6_K / Q8_0
  default_temperature: 0.2        # optional; overrides dialect default
  default_top_p: null             # optional; null keeps dialect default
---
```

## Field-by-field

### Top-level

| Field | Type | Default | Notes |
|---|---|---|---|
| `dlm_id` | 26-char ULID | required | Assigned by `dlm init`. Never regenerated. |
| `dlm_version` | int ≥ 1 | `1` | Bumped by `dlm migrate` when the schema evolves. |
| `base_model` | non-empty str | required | Registry key or `hf:org/name`. |
| `system_prompt` | str or null | null | Emitted as `SYSTEM "…"` in the Modelfile on export. |
| `training` | object | defaults | See below. |
| `export` | object | defaults | See below. |

### `training`

| Field | Type | Default | Notes |
|---|---|---|---|
| `adapter` | `lora` or `qlora` | `lora` | QLoRA requires CUDA + bitsandbytes. |
| `lora_r` | int 1..256 | 8 | LoRA rank. |
| `lora_alpha` | int ≥ 1 | 16 | LoRA alpha (scaling). |
| `lora_dropout` | float 0..0.5 | 0.05 | |
| `target_modules` | `auto` or list | `auto` | `auto` uses the per-architecture registry from Sprint 06. Explicit lists override. |
| `sequence_len` | int 64..32768 | 2048 | Max token length per example. Also emitted as Ollama `PARAMETER num_ctx`. |
| `micro_batch_size` | `auto` or int ≥ 1 | `auto` | Doctor picks based on VRAM. |
| `grad_accum` | `auto` or int ≥ 1 | `auto` | Doctor picks to reach effective batch = 8. |
| `learning_rate` | float > 0 | 2e-4 | |
| `num_epochs` | int ≥ 1 | 3 | |
| `optimizer` | enum | `adamw_torch` | `adamw_bnb_8bit` / `paged_adamw_8bit` for CUDA + bnb. |
| `lr_scheduler` | enum | `cosine` | |
| `warmup_ratio` | float 0..0.5 | 0.1 | |
| `precision` | `bf16` / `fp16` / `fp32` or null | null | Override the doctor's auto-pick. Defaults: bf16 on Ampere+/ROCm-bf16, fp16 on older CUDA, **fp32 on MPS** (the MPS fp16 attention kernels produce NaN LoRA weights on tiny-data SFT — see bug note below). Set `fp16` on MPS only if you need the memory headroom for a 7–8B base and your data isn't pathologically small; the post-train finite-weights gate will still refuse to persist a corrupt adapter. |
| `seed` | int | 42 | Determinism seed. Changing it invalidates the [determinism golden](../determinism.md). |
| `sources` | list[SourceDirective] or null | null | Declarative file-tree ingestion. Each entry is walked at train time; matching files become synthetic PROSE sections on the CPT path. See below. |
| `sources_policy` | `permissive` / `strict` | `permissive` | `strict` confines directive paths to the `.dlm`'s parent subtree; `permissive` allows absolute paths anywhere. Symlink escapes are refused under strict, warned under permissive. |
| `gate` | GateConfig | defaults | Learned MoE-style adapter gate (schema v8). See below. |

### `training.gate` — GateConfig

Learned adapter routing. A small MLP trained post-SFT that maps a
prompt embedding to per-adapter weights, replacing the hand-set
`--adapter-mix` for the `dlm prompt` path.

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `false` | Opt-in. Requires `training.adapters` with ≥2 named adapters. |
| `hidden_proj_dim` | int 8..2048 | `64` | Gate MLP internal width. Default is ~0.5MB for 4 adapters × 2048 hidden. |
| `steps` | int 1..10000 | `200` | AdamW iterations for the post-SFT gate training pass. |
| `lr` | float 0..1 | `3e-4` | AdamW learning rate. |
| `cold_start_floor` | int 1..1024 | `4` | Per-adapter minimum supervising sections. Below this, gate training is skipped and a uniform-mode `gate_config.json` is written instead. |
| `entropy_lambda` | float 0..1 | `0.01` | Shannon-entropy regularizer on the gate loss. Higher values discourage mode collapse; lower values let the gate commit harder. |

Enabling `gate` on a document without `training.adapters` (or with
only one adapter) is refused at parse time — a router over a single
adapter has nothing to route between. See
`docs/cookbook/learned-adapter-gate.md` for the full workflow +
Ollama-export fallback semantics.

### `training.sources[]` — SourceDirective

One entry per external root to ingest. Paths resolve relative to the
`.dlm` file's parent when not absolute; `~` expands to `$HOME`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `path` | non-empty str | required | File or directory path. Relative → anchored at the `.dlm`'s parent. |
| `include` | list[str] | `["**/*"]` | Glob patterns (POSIX, `**` spans directories). At least one must match for a file to be ingested. |
| `exclude` | list[str] | `[]` | Glob patterns evaluated first; any match drops the file. |
| `max_bytes_per_file` | int ≥ 1 or null | null | Files larger than this are skipped with one log line. |
| `max_files` | int ≥ 1 or null | null | Deterministic truncation: lexicographic-sorted walk keeps the first-N. |

Behavior:

- **File enumeration is deterministic.** Lexicographic sort on the
  resolved path list; identical trees on identical OSes produce
  identical Section order.
- **Binary files are skipped** (NUL byte in the first KiB — the
  standard grep heuristic). Skip count is recorded in the training
  summary.
- **UTF-8 decode failures are skipped**, not fatal. Use `exclude` for
  known-non-UTF-8 formats.
- **Each ingested file becomes a PROSE section** whose content is
  prefixed with `# source: <relpath>\n\n`. The path prefix ensures
  two files with identical bodies produce distinct `section_id`s —
  the replay corpus tracks per-file identity, not per-content.
- **Integration is seamless** with in-body sections. The CPT path,
  replay corpus, content-hash diff, and deterministic train/val
  split all treat directive-sourced sections identically.

Example:

```yaml
training:
  sources_policy: permissive
  sources:
    - path: ~/code/quillstone-protocol
      include: ["**/*.py", "**/*.rs"]
      exclude: ["tests/**", "**/__pycache__/**"]
      max_bytes_per_file: 65536
      max_files: 5000
    - path: ~/notes/research.md
```

After `dlm train`, the training summary JSON carries a
`source_directives: [...]` array with per-source file counts, byte
totals, and skip breakdowns. `dlm show --json` reports the same
under `training_sources`.

**Secrets warning:** directive ingestion has no implicit exclude
list. Add explicit `exclude: ["**/.env", "**/credentials*", ...]`
or use `sources_policy: strict` + a curated subtree to avoid
training on `.env`, private keys, or other sensitive files that
happen to live in your codebase.

### `export`

| Field | Type | Default | Notes |
|---|---|---|---|
| `default_quant` | `Q4_K_M`/`Q5_K_M`/`Q6_K`/`Q8_0` | `Q4_K_M` | Used when `dlm export --quant` isn't passed. |
| `default_temperature` | float 0..2 or null | null | Per-document sampling override. Emitted as Modelfile `PARAMETER temperature`. |
| `default_top_p` | float 0..1 or null | null | Per-document sampling override. |

## Migrations

When a new version bumps `dlm_version` (e.g., adding a field),
`dlm migrate` runs the registered migrators in order and rewrites the
frontmatter in place. See Sprint 12b for the migration framework.

The parser refuses to load a document whose `dlm_version` exceeds the
running CLI's `CURRENT_SCHEMA_VERSION`:

```
error: tutor.dlm:2:14 — dlm_version 2 is newer than this CLI supports (1).
       Upgrade dlm to continue.
```
