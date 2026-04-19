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

Bootstrap a new `.dlm` file with a fresh ULID.

```
dlm init <path> [--base <key>] [--i-accept-license]
```

| Option | Default | Notes |
|---|---|---|
| `--base <key>` | `smollm2-135m` | Registry key or `hf:org/name`. |
| `--i-accept-license` | false | Required for gated bases (Llama-3.2). |

Writes `<path>` with the minimum frontmatter + an empty body. Refuses
if the file already exists.

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
| `--fresh` | true | Discard prior optimizer state; train from scratch. Mutex with `--resume`. |
| `--seed N` | frontmatter.training.seed | Override training seed. |
| `--max-steps N` | unlimited | Cap step count. |
| `--i-accept-license` | false | Required for gated bases. |
| `--strict-lock` | false | Fail on any `dlm.lock` drift (even WARN). |
| `--update-lock` | false | Bypass validation; always write a fresh `dlm.lock`. |
| `--ignore-lock` | false | Bypass validation; don't write `dlm.lock`. |

The three lock flags are mutually exclusive. See [Determinism](../determinism.md)
for the mismatch severity table.

### `dlm prompt`

Run inference against the current adapter.

```
dlm prompt <path> [query] [--max-tokens N] [--temp F] [--top-p F]
                  [--verbose]
```

| Option | Default | Notes |
|---|---|---|
| `--max-tokens N` | 256 | Max new tokens to generate. |
| `--temp F` | 0.7 | Temperature. `0.0` = greedy decoding (deterministic). |
| `--top-p F` | None | Top-p sampling. |
| `--verbose` | false | Print resolved `InferencePlan` on stderr. |

Query is the CLI positional argument. Omit to read from stdin.

### `dlm export`

Produce GGUF files + Modelfile + register with Ollama.

```
dlm export <path> [--quant Q] [--merged [--dequantize]]
                  [--name N] [--skip-ollama] [--no-smoke] [--no-imatrix]
                  [--draft TAG | --no-draft]
                  [--adapter-mix name:w,...]
```

| Option | Default | Notes |
|---|---|---|
| `--quant Q` | frontmatter.export.default_quant | `Q4_K_M` / `Q5_K_M` / `Q6_K` / `Q8_0` / `F16`. |
| `--merged` | false | Merge LoRA into base before quantizing. |
| `--dequantize` | false | Required with `--merged` on a QLoRA adapter (pitfall #3). |
| `--name N` | derived | Ollama model name. |
| `--skip-ollama` | false | Emit GGUFs but don't register. |
| `--no-smoke` | false | Register but skip the smoke prompt. |
| `--no-imatrix` | false | Opt out of imatrix-calibrated quantization. |
| `--draft TAG` | auto | Override the speculative-decoding draft model. |
| `--no-draft` | false | Disable speculative decoding. Mutex with `--draft`. |
| `--adapter-mix name:w,...` | none | Multi-adapter export (Sprint 20). |

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

Every error path routes through the Rich reporter in
`dlm.cli.reporter` so failure messages are consistent across commands.
