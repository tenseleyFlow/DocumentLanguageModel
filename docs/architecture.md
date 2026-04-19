# Architecture

A compressed map of how DLM is organized. For the sprint-level
history, see `.docs/sprints/` in the repo (planning artifacts kept
local).

## The big idea

```
.dlm file  ──▶  parser ──▶  dataset builder ──▶  SFTTrainer  ──▶  LoRA adapter
   │                            ▲                                      │
   │                            │                                      ▼
   └──▶  replay corpus ─────────┘                                 GGUF + Modelfile
                                                                       │
                                                                       ▼
                                                                  ollama create
```

The `.dlm` source is the input; a trained LoRA adapter is the output.
Everything in between is opinionated engineering: content-addressed
storage, a determinism contract, a hardware doctor, an explicit Go
chat template, preflight checks against every footgun we've found.

## Module map

| Module | What it owns |
|---|---|
| `dlm.doc` | `.dlm` parser, serializer, Pydantic schema, section grammar. |
| `dlm.store` | Content-addressed store at `~/.dlm/store/<id>/`. Paths, manifest, exclusive lock, introspection. |
| `dlm.base_models` | Curated registry of launch-day bases; `hf:` escape hatch; compatibility probes; license acceptance. |
| `dlm.hardware` | Backend detection (CUDA / MPS / ROCm / CPU), capability probing, memory estimation, refusal matrix, `TrainingPlan` resolver. |
| `dlm.data` | Section → dataset row adapter, tokenizer bring-up (pad ≠ EOS rule), TRL formatting. |
| `dlm.replay` | Zstd-compressed append-only corpus + recency-weighted sampler + delta-against-manifest. |
| `dlm.train` | Orchestrator: preflight → determinism → load → train → two-phase commit → state sidecar → manifest update. |
| `dlm.eval` | Perplexity / val-loss callback + early-stop + training-summary writer. |
| `dlm.inference` | HF-heavy path for `dlm prompt`; `InferencePlan` resolver. |
| `dlm.export` | GGUF conversion, adapter GGUF, quantization, imatrix calibration, embedding-row sha, merge-safety gate. |
| `dlm.export.ollama` | Modelfile emission, Go template registry, `ollama create` + smoke, token-identity verification. |
| `dlm.pack` | `.dlm.pack` format (v1), packer, unpacker, integrity verification, migrations registry. |
| `dlm.lock` | Per-store `dlm.lock` schema, severity-table mismatch policy, validator, writer. |
| `dlm.cli` | Typer app + per-command glue; `dlm.cli.reporter` owns formatted error output. |
| `dlm.io` | `atomic` (write-and-rename), `text` (UTF-8 + LF normalization), `ulid`. |

## Storage layout

```
~/.dlm/store/<dlm_id>/
├── dlm.lock                       # Sprint 15 reproducibility contract
├── manifest.json                  # training runs + exports + content hashes
├── adapter/
│   ├── current.txt                # → versions/v0001
│   └── versions/
│       ├── v0001/
│       │   ├── adapter_config.json
│       │   ├── adapter_model.safetensors
│       │   ├── training_state.pt          # optimizer/scheduler/RNG
│       │   ├── training_state.pt.sha256
│       │   ├── training_run.json          # human-readable run metadata
│       │   └── pinned_versions.json
│       └── v0002/
├── replay/
│   ├── corpus.zst                 # append-only zstd-compressed section history
│   └── index.json
├── exports/
│   └── Q4_K_M/
│       ├── base.Q4_K_M.gguf
│       ├── adapter.gguf
│       ├── Modelfile
│       ├── export_manifest.json
│       └── imatrix.dat            # cached per-corpus-hash
├── cache/                         # scratch for convert scripts
└── logs/
    └── train-000001-*.jsonl       # per-step JSONL log
```

## Contract boundaries

Four load-bearing files; when editing, keep them distinct:

- **`manifest.json`** — running narrative of training runs, exports,
  and content hashes. Mutable on every run. Owned by Sprint 04.
- **`dlm.lock`** (per-store) — version pins + hardware tier +
  determinism flags + license acceptance. Owned by Sprint 15.
- **`training_state.pt`** — optimizer/scheduler/RNG for bit-exact
  resume. Owned by Sprint 09.
- **`exports/<quant>/export_manifest.json`** — per-export checksums,
  quant level, pinned llama.cpp tag, smoke output. Owned by Sprint 11.

## The determinism contract

Same `(.dlm source, base revision, hardware tier, pinned versions,
seed, determinism flags)` → same adapter SHA. Enforced by
`src/dlm/lock/` + the integration test under
`tests/integration/lock/test_determinism_golden.py`. See
[Determinism](determinism.md) for details.

## Sprint timeline

| Phase | Sprints | Release |
|---|---|---|
| 0 — Foundation | 01–05 (scaffolding → hardware doctor) | v0.1 |
| 1 — Core training | 06–10 (registry → replay → trainer → eval) | v0.5 |
| 2 — Export | 11–12 (+ 11.5, 11.6, 12.5, 12.6 follow-ups) | v0.8 |
| 3 — MVP release | 12b, 13, 14, 14.5, 15, 16 (this sprint) | **v1.0** |
| 4 — Advanced training | 17–20 (DPO, ORPO, CPT, multi-adapter) | v1.x |
| 5 — Performance & scale | 21–23 (MLX, ROCm, multi-GPU) | v1.x / v2 |
| 6 — UX polish | 24–26 (REPL, watch mode, observability) | v2 |
| 7 — Ecosystem | 27–28 (gallery, share protocol) | v2+ |

Every sprint has a binary Definition of Done; status snapshots live in
`.docs/sprints/00-index.md` in the repo (local-only by user choice).
