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
dlm init <path> [--base <key>] [--template <name>]
                [--multimodal | --audio]
                [--skip-export-probes]
                [--i-accept-license] [--force]
```

| Option | Default | Notes |
|---|---|---|
| `--base <key>` | `qwen2.5-1.5b` | Registry key or `hf:org/name`. Ignored when `--template` is used (the template's `recommended_base` wins). With `--multimodal`, defaults to `paligemma-3b-mix-224`. |
| `--template <name>` | None | Bootstrap from a named gallery template. See `dlm templates list`. Mutually exclusive with `--multimodal`. |
| `--skip-export-probes` | false | Skip the llama.cpp / GGUF compatibility probes so a brand-new architecture can still be used for training + HF inference. Forfeits `dlm export` to Ollama until the vendored exporter catches up. |
| `--multimodal` | false | Scaffold a vision-language `.dlm` with an `::image::` section (schema v10). Flips `--base` to `paligemma-3b-mix-224` unless explicitly overridden; a non-VL `--base` is refused. See [multimodal-training cookbook](../cookbook/multimodal-training.md). |
| `--audio` | false | Scaffold an audio-language `.dlm` with an `::audio::` section. Flips `--base` to `qwen2-audio-7b-instruct`, skips export probes, and refuses text / vision-language bases. See [audio-training cookbook](../cookbook/audio-training.md). |
| `--i-accept-license` | false | Required for gated bases (Llama-3.2, PaliGemma). |
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
                 [--phase {sft,preference,all}]
                 [--i-accept-license]
                 [--strict-lock|--update-lock|--ignore-lock]
                 [--strict-metrics]
                 [--base <key>] [--include GLOB]... [--exclude GLOB]...
                 [--recursive|--no-recursive] [--name NAME]
                 [--policy {strict,permissive}] [--rescaffold]
                 [--skip-export-probes]
```

| Option | Default | Notes |
|---|---|---|
| `--resume` | false | Continue from `training_state.pt`. Mutex with `--fresh`. |
| `--fresh` | false | Discard prior optimizer state; train from scratch. Mutex with `--resume`. Default when neither flag is set. |
| `--seed N` | frontmatter.training.seed | Override training seed. |
| `--max-steps N` | unlimited | Cap step count. |
| `--phase {sft,preference,all}` | `all` | Choose which training phases run: SFT only, preference only, or both in sequence. Preference-only requires a prior SFT adapter. |
| `--i-accept-license` | false | Required for gated bases (usually captured once at `dlm init` and persisted). |
| `--strict-lock` | false | Fail on any `dlm.lock` drift (even WARN). |
| `--update-lock` | false | Bypass validation; always write a fresh `dlm.lock`. |
| `--ignore-lock` | false | Bypass validation; don't write `dlm.lock`. |
| `--strict-metrics` | false | Promote metrics SQLite write failures to hard errors instead of best-effort degradation. Run-start / run-end are always hard-fail anchors; this flag extends that policy to step, eval, tokenization, and export streams. |
| `--gpus SPEC` | single-process | Multi-GPU training via Accelerate. `all` uses every visible CUDA device; `N` uses the first N; `0,1` selects exact device ids. Dispatches to `accelerate launch` when >1 device is selected. Refused on MPS/CPU/ROCm; heterogeneous CUDA SMs refused. |
| `--watch` | false | Save-to-train mode (Sprint 25). After the initial train, block on filesystem events and re-run bounded-step retrains on each settled save. |
| `--watch-max-steps N` | 100 | Per-cycle step cap for `--watch`. Keeps cycles responsive. |
| `--watch-debounce-ms N` | 400 | Quiet interval before a burst of saves triggers a retrain. |
| `--repl` | false | With `--watch`: also run `dlm repl` in the same process. **Scaffolded only** — threading integration is a followup; today the flag refuses with exit 2. |
| `--base <key>` | required on first auto-scaffold | Base model for `dlm train <dir>` auto-scaffold. Ignored when `<path>` already points at a `.dlm`. |
| `--include GLOB` | repeatable | Auto-scaffold include glob. Defaults to `**/*` with `--recursive`, `*` with `--no-recursive`. |
| `--exclude GLOB` | repeatable | Auto-scaffold exclude glob. Directory-descent defaults still apply on top. |
| `--recursive` / `--no-recursive` | recursive | Auto-scaffold whether default include globs descend into subdirectories. |
| `--name NAME` | `corpus` | Auto-scaffold target file name under `<dir>/.dlm/<name>.dlm`. Lets one tree host multiple adapters. |
| `--policy {strict,permissive}` | `strict` | Auto-scaffold `training.sources_policy`. `strict` confines training sources to the target directory; `permissive` allows absolute paths anywhere. |
| `--rescaffold` | false | Rewrite an existing scaffolded `.dlm` in place with new auto-scaffold flags while preserving its `dlm_id`. |
| `--no-cache` | false | Bypass the tokenized-section cache for this run. Default is cache-on when the `.dlm` declares `training.sources`. Use when debugging tokenization or cross-checking cached-vs-uncached determinism. Entries from prior runs stay on disk; the next run without the flag picks them back up. See [directive-cache](../cookbook/directive-cache.md). |
| `--skip-export-probes` | false | Skip the llama.cpp / GGUF compatibility probes so a brand-new architecture can still be trained for HF inference. Mirrors `dlm init --skip-export-probes`. |

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
                  [--adapter NAME] [--gate {auto,off}]
                  [--image PATH]... [--audio PATH]...
                  [--verbose]
```

| Option | Default | Notes |
|---|---|---|
| `--max-tokens N` | 256 | Max new tokens to generate. |
| `--temp F` | 0.7 | Temperature. `0.0` = greedy decoding (deterministic). |
| `--top-p F` | None | Top-p sampling. |
| `--adapter NAME` | None | Select a named adapter from `training.adapters`. Required on multi-adapter documents; rejected on single-adapter ones. |
| `--gate {auto,off}` | `auto` | Learned adapter gate (Sprint 34). `auto` uses the trained gate when one exists in the store; `off` forces uniform weights across declared adapters. Silently ignored when `--adapter` pins a single adapter. See `docs/cookbook/learned-adapter-gate.md`. |
| `--image PATH` | none | Attach an image to the prompt. Repeat for multiple images; each expands to one `<image>` placeholder the processor slots pixels into. Required on vision-language bases; rejected on text bases. See [multimodal-training cookbook](../cookbook/multimodal-training.md). |
| `--audio PATH` | none | Attach an audio clip to the prompt. Repeat for multiple clips. Required on audio-language bases; rejected on text and vision-language bases. See [audio-training cookbook](../cookbook/audio-training.md). |
| `--backend {auto,pytorch,mlx}` | `auto` | Inference backend. `auto` picks MLX on Apple Silicon (when `uv sync --extra mlx` is installed), else PyTorch. Ignored on VL bases (the VL path always uses PyTorch + AutoModelForImageTextToText). |
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
dlm metrics <path> watch [--poll-seconds N]
```

| Option | Default | Notes |
|---|---|---|
| `--json` | false | Emit JSON object (`{runs: [...], steps: [...], evals: [...]}` when combined with `--run-id`). |
| `--csv` | false | Emit CSV of runs or (with `--run-id`) steps + evals. |
| `--run-id N` | None | Drill into one run; prints its step/eval counts. |
| `--phase` | None | Filter runs by phase (`sft`/`dpo`/`orpo`/`cpt`). |
| `--since` | None | Time window (`24h`, `7d`, `30m`, `10s`). |
| `--limit N` | 20 | Cap the number of runs returned. |

`dlm metrics <path> watch` polls the DB and tails new step/eval rows as
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

Produce GGUF files + runtime-target metadata.

```
dlm export <path> [--target NAME] [--quant Q] [--merged [--dequantize]]
                  [--name N] [--no-template] [--skip-ollama]
                  [--no-smoke] [--no-imatrix] [--verbose]
                  [--draft TAG | --no-draft]
                  [--adapter NAME | --adapter-mix SPEC]
```

| Option | Default | Notes |
|---|---|---|
| `--target NAME` | `ollama` | Export destination. Sprint 41 currently supports `ollama` and `llama-server`. The `llama-server` path writes launch artifacts against the existing GGUF export and currently requires `--no-smoke` while the HTTP smoke harness lands. |
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

### `dlm verify`

Verify a `.dlm.pack` provenance chain before trusting or installing it.

```
dlm verify <pack> [--trust-on-first-use]
```

| Option | Default | Notes |
|---|---|---|
| `--trust-on-first-use` | false | Record an unknown signer's public key into `~/.dlm/trusted-keys/` on first verify. Without it, unknown signers are refused with exit code 2. |

Exit codes: `0` verified, `1` broken chain or missing provenance,
`2` untrusted signer, `3` signature rejected.

### `dlm push`

Upload a `.dlm` (auto-packs) or `.dlm.pack` to a sharing destination
(Sprint 28).

```
dlm push <path> --to <destination> [--sign] [pack flags]
```

| Option | Default | Notes |
|---|---|---|
| `--to <destination>` | required | `hf:<org>/<repo>`, `https://...` URL endpoint, or a local path. |
| `--sign` | false | Sign the pack with `minisign` before upload (requires `minisign` on PATH + key at `~/.dlm/minisign.key`). |
| `--include-exports` | false | Forwarded to `dlm pack` when auto-packing a `.dlm`. |
| `--include-base` | false | Same. |
| `--include-logs` | false | Same. |
| `--i-am-the-licensee URL` | none | Required with `--include-base` on a non-redistributable base. |

**Destinations:**
- `hf:<org>/<repo>` — HuggingFace Hub. Uses `$HF_TOKEN` if set. Autogenerates a `README.md` with `library_name: dlm` tag. Creates the repo if missing (your personal namespace needs no approval).
- `https://…` — any HTTPS endpoint that accepts a POST with an `application/octet-stream` body. Sets `Authorization:` from `$DLM_SHARE_AUTH` when present (e.g. `Bearer <token>`).
- `<local/path>` — copy the pack to a filesystem path.

### `dlm pull`

Download + verify + unpack a `.dlm.pack` from a remote source.

```
dlm pull <source> [--out DIR] [--force]
```

| Option | Default | Notes |
|---|---|---|
| `<source>` | required | `hf:<org>/<repo>`, `https://…`, `peer://host:port/<id>?token=…`, or a local path. |
| `--out DIR` | CWD | Directory for the restored `.dlm`. |
| `--force` | false | Overwrite an existing store with the same `dlm_id`. |

Pulls always verify sha256 checksums during unpack. If a `.minisig`
sidecar is served alongside the pack, `dlm pull` tries every key in
`~/.dlm/trusted-keys/*.pub` — match → `verified`, no match →
`unverified` warning (still installs, checksums are fine). No sidecar
→ `unsigned` (still installs).

### `dlm serve`

Serve a `.dlm`'s pack over LAN for peers to pull.

```
dlm serve <path> [--port N] [--public --i-know-this-is-public]
                 [--max-concurrency N] [--rate-limit N]
                 [--token-ttl-minutes N]
```

| Option | Default | Notes |
|---|---|---|
| `--port N` | 7337 | Bind port. |
| `--public` | false | Bind `0.0.0.0` **only when paired with** `--i-know-this-is-public`. Without the confirmation flag, `--public` logs a refusal and binds `127.0.0.1`. |
| `--i-know-this-is-public` | false | Acknowledges the public bind. Meaningless without `--public`. |
| `--max-concurrency N` | 4 | Max concurrent connections per token. Excess returns HTTP 429. |
| `--rate-limit N` | 30 | Max requests per minute per token. |
| `--token-ttl-minutes N` | 15 | Issued token lifetime. Ctrl-C invalidates every outstanding token instantly — the session secret lives only in the serving process. |

On start, prints the `peer://` URL (with embedded token) that the
other side pastes into `dlm pull`. Ctrl-C cleanly stops the server
and deletes the temp pack.

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

### `dlm cache`

Inspect and manage the per-store tokenized-section cache (Sprint 31).
The cache speeds up re-training on directive-sourced codebases by
keying tokenized output on `(section_id, tokenizer_sha, sequence_len)`.

```
dlm cache show <path> [--json]
dlm cache prune <path> [--older-than DURATION]
dlm cache clear <path> [--force]
```

| Subcommand | Notes |
|---|---|
| `show` | Print entry count, size on disk, last-run hit rate. `--json` for machine-readable output. |
| `prune` | Delete entries not accessed within `--older-than` (e.g. `30d`, `12h`, `45m`). Default `90d`. |
| `clear` | Wipe the entire cache. Prompts for confirmation unless `--force` is passed. |

See `docs/cookbook/directive-cache.md` for tuning, invalidation
triggers, and maintenance patterns.

### `dlm harvest`

Pull failing-probe results from a sway-style eval report back into the
document as `!probe`-tagged `::instruction::` sections for the next
retrain. See `docs/cookbook/probe-driven-training.md`.

```
dlm harvest <path> --sway-json <report> [--apply] [--dry-run]
                   [--tag NAME] [--min-confidence F]
                   [--strict | --lax]
dlm harvest <path> --revert
```

| Option | Default | Notes |
|---|---|---|
| `--sway-json PATH` | required | Path to the sway probe report JSON. |
| `--apply` | false | Write changes to disk. Without it, dry-run. |
| `--dry-run` | true | Print the diff; no writes. |
| `--revert` | — | Strip all `auto_harvest=True` sections; mutually exclusive with `--sway-json`. |
| `--tag NAME` | `auto-harvest` | Provenance tag written into `harvest_source`. |
| `--min-confidence F` | `0.0` | Skip candidates below this confidence. |
| `--strict` / `--lax` | lax | Strict: fail if any failing probe lacks a reference. Lax: skip + log. |

Exit codes: `0` success, `1` validation error (malformed JSON, strict
miss, mutual-exclusion violation), `2` no candidates to harvest.

### `dlm train --listen-rpc`

During `--watch`, open a JSON-RPC endpoint that accepts `inject_probe`
pushes from external eval harnesses. Requires `DLM_PROBE_TOKEN` in the
environment. See `docs/cookbook/probe-driven-training.md` for the wire
protocol and security notes.

| Option | Default | Notes |
|---|---|---|
| `--listen-rpc HOST:PORT` | off | Bind the probe-RPC endpoint. Requires `--watch` or `--max-cycles`. |
| `--max-cycles N` | `0` | Bounded-loop alternative to `--watch` for convergence runs (scaffolded — currently refuses execution without `--watch`). |

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
