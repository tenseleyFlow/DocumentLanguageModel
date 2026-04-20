# CLI reference

Generated from the running `dlm --help` output. Auto-regeneration via
`typer-cli` is planned for a follow-up sprint; until then this file is
hand-maintained and gated by the test suite.

## Global options

Applied to every subcommand:

| Option | Env var | Default | Description |
|---|---|---|---|
| `--home PATH` | `DLM_HOME` | `~/.dlm` | Override the store root. |
| `-v, --verbose` | — | off | Emit plan / resolver diagnostics on stderr. |
| `-q, --quiet` | — | off | Suppress informational output. |
| `--version` | — | — | Print version and exit. |
| `--install-completion` | — | — | Install shell completion. |
| `--show-completion` | — | — | Print shell completion script. |
| `-h, --help` | — | — | Show command help. |

## Commands

### `dlm init`

Bootstrap a new `.dlm` file with a fresh ULID, create the per-store
directory, and persist the license-acceptance record (audit-05 B2).

```
dlm init <path> [--base <key>] [--template <name>] [--i-accept-license] [--force]
```

| Option | Default | Notes |
|---|---|---|
| `--base <key>` | `qwen2.5-1.5b` | Registry key or `hf:org/name`. Ignored when `--template` is used (the template's `recommended_base` wins). |
| `--template <name>` | None | Bootstrap from a named gallery template. See `dlm templates list`. |
| `--i-accept-license` | false | Required for gated bases (Llama-3.2). |
| `--force` | false | Overwrite an existing `.dlm` at path. |

Writes `<path>` with minimum frontmatter, provisions
`~/.dlm/store/<dlm_id>/` with an initial `manifest.json`, and (for
gated bases) stores the `LicenseAcceptance` record so `dlm train` /
`dlm export` don't re-prompt. Refuses if the `.dlm` file already
exists and `--force` wasn't passed.

### `dlm train`

Train / retrain the adapter.

```
dlm train <path> [--resume|--fresh] [--seed N] [--max-steps N]
                 [--i-accept-license]
                 [--strict-lock|--update-lock|--ignore-lock]
```

| Option | Default | Notes |
|---|---|---|
| `--resume` | false | Continue from `training_state.pt`. Mutex with `--fresh`. |
| `--fresh` | false | Discard prior optimizer state; train from scratch. Mutex with `--resume`. Default when neither flag is set. |
| `--seed N` | frontmatter.training.seed | Override training seed. |
| `--max-steps N` | unlimited | Cap step count. |
| `--i-accept-license` | false | Required for gated bases (usually captured once at `dlm init` and persisted). |
| `--strict-lock` | false | Fail on any `dlm.lock` drift (even WARN). |
| `--update-lock` | false | Bypass validation; always write a fresh `dlm.lock`. |
| `--ignore-lock` | false | Bypass validation; don't write `dlm.lock`. |
| `--gpus SPEC` | single-process | Multi-GPU training via Accelerate. `all` uses every visible CUDA device; `N` uses the first N; `0,1` selects exact device ids. Dispatches to `accelerate launch` when >1 device is selected. Refused on MPS/CPU/ROCm; heterogeneous CUDA SMs refused. |
| `--watch` | false | Save-to-train mode (Sprint 25). After the initial train, block on filesystem events and re-run bounded-step retrains on each settled save. |
| `--watch-max-steps N` | 100 | Per-cycle step cap for `--watch`. Keeps cycles responsive. |
| `--watch-debounce-ms N` | 400 | Quiet interval before a burst of saves triggers a retrain. |
| `--repl` | false | With `--watch`: also run `dlm repl` in the same process. **Scaffolded only** — threading integration is a followup; today the flag refuses with exit 2. |

The three lock flags are mutually exclusive. See [Determinism](../determinism.md)
for the mismatch severity table.

`--gpus` multiplies the effective batch size by `world_size`; the
resulting lock records `world_size` and warns on drift between runs.
Multi-GPU + QLoRA on CUDA is permitted (bitsandbytes supports DDP);
multi-GPU + ROCm is out of scope for Sprint 23.

### `dlm prompt`

Run inference against the current adapter.

```
dlm prompt <path> [query] [--max-tokens N] [--temp F] [--top-p F]
                  [--adapter NAME] [--verbose]
```

| Option | Default | Notes |
|---|---|---|
| `--max-tokens N` | 256 | Max new tokens to generate. |
| `--temp F` | 0.7 | Temperature. `0.0` = greedy decoding (deterministic). |
| `--top-p F` | None | Top-p sampling. |
| `--adapter NAME` | None | Select a named adapter from `training.adapters`. Required on multi-adapter documents; rejected on single-adapter ones. |
| `--backend {auto,pytorch,mlx}` | `auto` | Inference backend. `auto` picks MLX on Apple Silicon (when `uv sync --extra mlx` is installed), else PyTorch. |
| `--verbose` | false | Print resolved `InferencePlan` on stderr. |

Query is the CLI positional argument. Omit to read from stdin.

### `dlm repl`

Interactive prompt-and-respond REPL against the trained adapter
(Sprint 24).

```
dlm repl <path> [--adapter NAME] [--backend {auto,pytorch,mlx}]
```

| Option | Default | Notes |
|---|---|---|
| `--adapter NAME` | None | Named adapter; required on multi-adapter docs. |
| `--backend {auto,pytorch,mlx}` | `auto` | Same contract as `dlm prompt --backend`. |

Slash commands inside the REPL: `/help`, `/exit`, `/clear`, `/save`,
`/adapter`, `/params`, `/model`, `/history`. Ctrl-D exits; Ctrl-C
cancels generation or input. Session history persists at
`~/.dlm/history`. See the [interactive-session cookbook](../cookbook/interactive-session.md).

### `dlm metrics`

Query the per-store SQLite metrics DB (Sprint 26).

```
dlm metrics <path> [--json|--csv] [--run-id N] [--phase PHASE] [--since WINDOW] [--limit N]
dlm metrics watch <path> [--poll-seconds N]
```

| Option | Default | Notes |
|---|---|---|
| `--json` | false | Emit JSON object (`{runs: [...], steps: [...], evals: [...]}` when combined with `--run-id`). |
| `--csv` | false | Emit CSV of runs or (with `--run-id`) steps + evals. |
| `--run-id N` | None | Drill into one run; prints its step/eval counts. |
| `--phase` | None | Filter runs by phase (`sft`/`dpo`/`orpo`/`cpt`). |
| `--since` | None | Time window (`24h`, `7d`, `30m`, `10s`). |
| `--limit N` | 20 | Cap the number of runs returned. |

`dlm metrics watch` polls the DB and tails new step/eval rows as
they arrive. See the [metrics cookbook](../cookbook/metrics.md) for
the full flow + optional TensorBoard / W&B sinks (`uv sync --extra
observability`).

### `dlm templates`

Browse the starter template gallery (Sprint 27).

```
dlm templates list [--json] [--refresh] [--accept-unsigned]
```

| Option | Default | Notes |
|---|---|---|
| `--json` | false | Emit the full `TemplateMeta` for each entry as JSON. |
| `--refresh` | false | Refresh from the upstream gallery. **Currently a no-op** — upstream repo and signing key are pending (Sprint 27 deferred polish); the command warns and falls back to the bundled gallery. |
| `--accept-unsigned` | false | Reserved. Will bypass signed-tag verification once the live fetcher is wired. |

Pair with `dlm init --template <name>` to create a new `.dlm`:

```bash
dlm init mydoc.dlm --template coding-tutor
```

See the [template-gallery cookbook](../cookbook/template-gallery.md)
for the full walkthrough and the `TemplateMeta` schema.

### `dlm export`

Produce GGUF files + Modelfile + register with Ollama.

```
dlm export <path> [--quant Q] [--merged [--dequantize]]
                  [--name N] [--no-template] [--skip-ollama]
                  [--no-smoke] [--no-imatrix] [--verbose]
                  [--draft TAG | --no-draft]
                  [--adapter NAME | --adapter-mix SPEC]
```

| Option | Default | Notes |
|---|---|---|
| `--quant Q` | frontmatter.export.default_quant | `Q4_K_M` / `Q5_K_M` / `Q6_K` / `Q8_0` / `F16`. |
| `--merged` | false | Merge LoRA into base before quantizing. |
| `--dequantize` | false | Required with `--merged` on a QLoRA adapter (pitfall #3). |
| `--name N` | derived | Ollama model name. |
| `--no-template` | false | Skip writing `TEMPLATE` into the Modelfile (power users only — Ollama will fuzzy-match, which Sprint 12 deliberately works around). |
| `--skip-ollama` | false | Emit GGUFs but don't register. |
| `--no-smoke` | false | Register but skip the smoke prompt. |
| `--no-imatrix` | false | Opt out of imatrix-calibrated quantization. |
| `--verbose` | false | Surface preflight + conversion diagnostics. |
| `--draft TAG` | auto | Override the speculative-decoding draft model. |
| `--no-draft` | false | Disable speculative decoding. Mutex with `--draft`. |
| `--adapter NAME` | None | Export a single named adapter from `training.adapters`. Rejected on single-adapter documents. Mutex with `--adapter-mix`. |
| `--adapter-mix SPEC` | None | Weighted composition like `knowledge:1.0,tone:0.5`. Produces one Ollama model by merging the named adapters at export time. LoRA-only; QLoRA sources require `--dequantize --merged`. Mutex with `--adapter`. |
| `--adapter-mix-method` | `linear` | PEFT merge strategy: `linear` (default; fast weighted sum) or `svd` (higher fidelity, heavier compute). Only meaningful with `--adapter-mix`. |

### `dlm pack`

Produce a portable `.dlm.pack` bundle.

```
dlm pack <path> [--out PATH] [--include-exports] [--include-base]
                [--include-logs] [--i-am-the-licensee URL]
```

| Option | Default | Notes |
|---|---|---|
| `--out PATH` | `<name>.dlm.pack` | Pack output. |
| `--include-exports` | false | Bundle all GGUF exports. |
| `--include-base` | false | Bundle the base model weights. Requires license acknowledgement for gated bases. |
| `--include-logs` | false | Bundle per-run JSONL logs. |
| `--i-am-the-licensee URL` | none | URL acknowledging separate base-license acceptance. |

### `dlm unpack`

Install a `.dlm.pack` into the local store.

```
dlm unpack <pack> [--force] [--out DIR]
```

| Option | Default | Notes |
|---|---|---|
| `--force` | false | Overwrite an existing store with the same `dlm_id`. |
| `--out DIR` | pack parent | Where to place the restored `.dlm`. |

### `dlm doctor`

Inspect hardware + print the resolved training plan.

```
dlm doctor [--json]
```

No-network. Probes torch + psutil only; refuses to go online.

### `dlm show`

Show training history, exports, and adapter state for a document.

```
dlm show <path> [--json]
```

Pretty-prints manifest + lock state. `--json` emits machine-readable
output.

### `dlm migrate`

Migrate a `.dlm` frontmatter to the current schema version.

```
dlm migrate <path> [--dry-run] [--no-backup]
```

| Option | Default | Notes |
|---|---|---|
| `--dry-run` | false | Print the migrated frontmatter without writing. |
| `--no-backup` | false | Skip the `.dlm.bak` backup. |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | Runtime failure (license refused, disk full, OOM, template drift, lock validation). |
| 2 | CLI misuse (mutex violation, missing argument). |

Domain errors are formatted consistently via bare `console.print`
calls in each subcommand (prefix convention: `<subject>: <message>`,
e.g. `lock: base_model_revision changed`). Uncaught exceptions escape
into `dlm.cli.reporter` which picks a matching prefix from the
module the exception came from and renders a tier-3 generic message.
