# Changelog

All notable changes to DocumentLanguageModel are recorded here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project targets [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **`dlm metrics` is now a subcommand group with an explicit `show`.**
  The previous shape — a callback that took `<path>` plus a `watch`
  subcommand — caused click to error with "Missing argument 'PATH'"
  whenever an option followed the positional (`dlm metrics PATH
  --run-id 1`). The fix makes the call explicit:
  `dlm metrics show PATH [options]`. `dlm metrics watch PATH` is
  unchanged. Update scripts that called the old form.

### Fixed

- **`gguf_arch` preflight probe was silently false-negative on every
  HF off-registry base.** Three compounding bugs surfaced while
  trying to train against `hf:Qwen/Qwen3-1.7B`:
  1. The probe's regex matched `@Model.register(...)` but upstream
     llama.cpp renamed the decorator to `@ModelBase.register(...)`
     mid-2024; the regex now accepts both forms.
  2. The regex captured only the *first* quoted arg, silently
     missing multi-arg decorators like
     `@ModelBase.register("Qwen3ForCausalLM", "Qwen3Model")`; the
     probe now extracts every quoted string inside the decorator's
     arg list.
  3. The probe compared `spec.gguf_arch` (short label like
     `"qwen3"`) against the decorator's arguments, but llama.cpp
     registers HF class names (`"Qwen3ForCausalLM"`) — different
     namespaces, will never match. Comparison now uses
     `spec.architecture`. The bug was invisible because registered
     models bypass the probe entirely; it only bit `hf:` paths.

### Added

- **`dlm export --emit-sway-json`** writes a ready-to-run
  `<export-dir>/sway.yaml` alongside the GGUF/Modelfile, eliminating
  the previous two-step `dlm export` → `sway autogen` ritual users
  had to do before evaluating an adapter via [sway](https://github.com/tenseleyFlow/sway).
  Calls into `dlm_sway.integrations.dlm.autogen.build_spec_dict` via
  a new `dlm.export.sway_json.write_sway_json` helper. Closes the X1
  half of sway's Sprint 26 cross-repo integration; X3 (sway-side
  `sway pack` / `sway unpack`) ships in sway proper.
  - New `[sway]` optional extra (`pip install 'dlm[sway]'`) pulls
    `dlm-sway>=0.1.0`. Deliberately pulls plain `dlm-sway`, NOT
    `dlm-sway[dlm]`, because the round-trip extra would create a
    pip-resolver cycle (sway's `[dlm]` extra already pulls dlm).
  - Failures route through a new typed `SwayJsonExportError`
    (subclass of `ExportError`) so the CLI's existing exception
    handler renders them cleanly. The most common failure — user
    didn't install the `[sway]` extra — gets a message that names
    the install command verbatim.
  - 5 unit tests in `tests/unit/cli/test_export_sway_json.py`
    cover the helper round-trip, missing-extra error, autogen
    failure wrapping, and CLI flag wiring.
- **`dlm train --skip-export-probes`** mirrors the flag on `dlm init`
  (it was missing from the train CLI; a user could `dlm init
  --skip-export-probes` a fresh .dlm then have `dlm train` re-run
  the probes and fail). The flag threads into `resolve_base_model`
  identically on both paths; help text matches verbatim.

## [0.10.0] — 2026-04-21

Four phases of work in a single release: advanced training, expanded
hardware coverage, the UX layer, and the ecosystem layer. 265 commits
since v0.9.0, five additive schema migrations (v1 → v6), six brutal
audits (audit-04 through audit-09) with remediations landed inline.

Still below 1.0 on purpose — the milestone for the semantic bump
remains the same as stated at v0.9.0: a human has to train + export +
`ollama run` a real document end-to-end and walk away satisfied. This
release is a broad feature expansion, not a stability claim.

### Breaking changes

None at the data level — schema migrations v1 → v2 → v3 → v4 → v5 →
v6 are all additive and run automatically via `dlm migrate`. Existing
`.dlm` files parse without modification.

One subtle CLI contract change: `dlm serve` now refuses an untrained
`.dlm` with an actionable error instead of a low-level
`ManifestCorruptError`. Scripts that relied on the previous behavior
exit-coded identically (exit 1) but must read the new message text.

### Advanced training

Preference tuning (Sprint 17) and its orchestration (Sprint 18):

- `::preference::` section fences with `### Prompt` / `### Chosen` /
  `### Rejected` grammar.
- DPO via TRL's `DPOTrainer`, ORPO via `trl.experimental.orpo`.
- `training.preference` frontmatter block (method / β / reference
  mode / loss type / max lengths).
- Phase orchestrator runs SFT → DPO/ORPO in sequence when preference
  content is present; `--phase sft|preference|all` overrides.
- Replay corpus gains `sample_preference_rows` — preference sections
  sample with the same recency-weighted reservoir as CPT rows.
- Doctor halves the micro-batch estimate and scales VRAM estimates
  when a DPO phase is active.

Continued pretraining refinements for DAPT (Sprint 19):

- `training.cpt` schema block — `CosineWithFloor` LR schedule,
  embed-layer warm-up, mixed-mode loss split reporting, vocab-gap
  diagnostics.
- Embed-layer freeze/unfreeze context manager wrapping the first N
  steps so vocab extensions settle before the backbone moves.
- Training summary adds per-mode loss fields so DAPT runs report SFT
  vs CPT loss separately.

Multi-adapter (Sprint 20a-c):

- `training.adapters: [name, config]` with mutual exclusion against
  the flat LoRA knobs.
- `dlm train --adapter <name>` / `dlm prompt --adapter <name>` /
  `dlm export --adapter <name>`.
- `dlm export --adapter-mix a:0.5,b:0.5` — weighted merge via
  `PEFT.add_weighted_adapter`, with QLoRA safety gate.
- Per-adapter store layout: `adapter/{name}/versions/vNNNN/`.
- Finite-weight and finite-eval gates — a training run that produces
  NaN weights or loss is rejected (renamed `-rejected`) instead of
  committed.
- `training.precision` override (schema v5) lets a document override
  the doctor's precision pick; MPS fp16 warns and pins to fp32 after
  a real NaN reproduction.

### Hardware

MLX inference backend (Sprint 21):

- PEFT safetensors → MLX `.npz` converter preserving adapter config.
- `MlxBackend` implementing the `InferenceBackend` protocol.
- `--backend mlx` flag on `dlm prompt`; doctor reports MLX
  availability.

ROCm training (Sprint 22):

- Tier-2 AMD GPU support via ROCm's HIP.
- bf16 + FlashAttention probes adapted for AMD.
- Custom llama.cpp ROCm build script.
- QLoRA-on-ROCm refusal with a precise error message.

Multi-GPU training (Sprint 23):

- `dlm train --gpus all|N|0,1` dispatches to `accelerate launch`.
- `rank_io.master_only` gates all trainer I/O so ranks don't
  duplicate writes.
- `DlmLock` gains `world_size` + `accelerate_version` fields for
  reproducibility.
- Doctor's effective-batch-size math respects the selected rank
  count.

### UX

Interactive REPL (Sprint 24):

- `dlm repl <path>` — `prompt_toolkit` loop against the trained
  adapter.
- Slash-command parser: `/seed`, `/temp`, `/top_p`, `/max_tokens`,
  `/system`, `/reload`, `/quit`.
- Persistent per-store history file.

Save-to-train watch mode (Sprint 25):

- `dlm train --watch` — `watchfiles` wrapper with debounced retrain
  on settled saves.
- Rich live status line (step, loss, elapsed, files watched).
- Ctrl-C exits cleanly between cycles.
- `--watch --repl` bridge is honestly deferred (marked `[~]` pending
  a CI-capable test harness).

Observability (Sprint 26):

- Per-store SQLite metrics database at
  `~/.dlm/store/<id>/metrics.db`.
- Typed event dataclasses: `RunStart`, `Step`, `Eval`, `RunEnd`,
  `TokenizationEvent`.
- `dlm metrics [--json|--csv]` — runs summary with filters.
- `dlm metrics watch <path>` — live tail of steps + evals.
- Optional sinks: TensorBoard (`[tb]` extra), W&B (`[wandb]` extra).

### Ecosystem

Template gallery (Sprint 27):

- `dlm templates list` — eight bundled templates (coding tutor,
  domain KB, writing partner, personal assistant, meeting notes,
  regex buddy, shell one-liner, study guide).
- `meta.yaml` sidecars per template (title, summary, recommended
  base, tags, license).
- `dlm init --template <name>` — fresh ULID, adopts the template's
  recommended base, persists license acceptance for gated bases.
- Offline-first registry; `--refresh` reserved for a future upstream
  gallery.

Share protocol (Sprint 28):

- `dlm push --to hf:org/name | https://... | peer://host:port`.
- `dlm pull <source>` with signature verification on peer and URL
  pulls.
- `dlm serve <path>` — LAN-local peer endpoint with HMAC bearer
  tokens, per-token rate limit, explicit public-bind gate.
- Optional minisign signing — sidecar `.minisig` next to the pack.
- HuggingFace Hub sink auto-generates a README from the manifest.

Source directives (Sprint 29):

- `training.sources: [...]` — declare file or directory sources in
  frontmatter; the trainer descends the tree and ingests raw text
  through the existing CPT path. `include` / `exclude` glob filters,
  per-file and per-source size/count caps.
- `sources_policy: permissive | strict` — strict confines paths to
  descendants of the `.dlm`'s directory with a symlink-escape check.
- Deterministic lexicographic enumeration; UTF-8 hygiene; binary
  detection via NUL sniff.
- Per-directive provenance in `TrainingRunSummary.source_directives`
  (file count, byte total, skip reasons).

`.dlm/` descent + auto-scaffold (Sprint 30):

- Per-codebase `.dlm/training.yaml` + `.dlm/ignore` discovered on a
  directory walk; nearest-ancestor resolution with gitignore-subset
  last-match-wins semantics (`!` negation, anchored `/`, trailing
  `/`, globstar `**`).
- Default-exclude set for VCS, caches, lockfiles, binaries.
- `Section.tags` flow from config metadata onto synthesized sections
  (loss weighting deferred to a future release).
- `dlm train <dir>` auto-scaffolds `<dir>/.dlm/corpus.dlm` on first
  run: ULID minted, `--base` + `--include` + `--exclude` + `--policy`
  baked in. Second invocation reuses the anchor.
- `--rescaffold` rewrites the scaffolded `.dlm` in place while
  preserving `dlm_id`.

Tokenized-section cache (Sprint 31):

- Per-store cache at `~/.dlm/store/<id>/tokenized-cache/`, keyed by
  `(section_id, tokenizer_sha256, sequence_len)`.
- Atomic tmp+rename writes, LRU eviction with current-run
  protection, tokenizer-version invalidation on SHA bump.
- `dlm cache show | prune | clear` CLI.
- **Deferred:** trainer-side wiring into the SFTTrainer tokenization
  path requires pre-tokenization plus a custom collator (label-shift
  preservation is subtle). Module is shipped and unit-tested; the
  consumer lands in a future release. See
  `src/dlm/directives/cache.py` module docstring.

### Audits + remediations

Six brutal audits ran during this window, each producing a
findings doc under `.docs/audits/` and remediation commits
referencing the finding IDs:

- Audit 04: replay-store integration, version-drift detection,
  tokenizer probe rename.
- Audit 05: pyproject runtime deps, license-acceptance record
  persistence, lock policy rules.
- Audit 06: 16 findings across GGUF parser hardening, ollama smoke
  tests, timezone-aware timestamps, pack hash determinism, vendor
  path resolution.
- Audit 07: forward-date schema rejection, ruff src-side cleanup.
- Audit 08: multi-GPU world_size plumbing, MLX adapter config fidelity,
  llama-cpp build env honoring, CLI reference drift.
- Audit 09 (Phase 7 brutal): `dlm train <dir>` end-to-end crash
  (B1 + B2), test-masks-bugs pattern (B3), orphan tokenized-cache
  (M1+M2 documented deferral), `dlm serve` guard on untrained `.dlm`
  (M3), task-tracker drift (M4), seven minors + two polish. Empirical
  differential evidence in `09-sway-appendix.md` — 359σ delta_kl vs
  null-adapter baseline on Fortran-idiomatic prompts.

### Schema migrations

All additive. Identity migrators; no data loss.

| From | To | Added                                       |
|------|----|---------------------------------------------|
| v1   | v2 | `training.preference` (DPO/ORPO) rename     |
| v2   | v3 | `training.cpt` block (schedule + warm-up)   |
| v3   | v4 | `training.adapters` (named multi-adapter)   |
| v4   | v5 | `training.precision` override               |
| v5   | v6 | `training.sources` + `sources_policy`       |

### New CLI surface

```
dlm templates list
dlm init --template <name>
dlm push --to <hf:...|https:...|peer://...> [--sign]
dlm pull <source>
dlm serve <path> [--public --i-know-this-is-public]
dlm repl <path>
dlm train --watch
dlm metrics [--json|--csv]
dlm metrics watch <path>
dlm train <dir> --base <key> --include <glob>
dlm cache show | prune | clear
```

### Test matrix

- 2,211 unit tests pass (≥95 % coverage on touched packages).
- ruff clean; mypy `--strict` clean across 215 source files.
- Slow integration matrix: two-adapter training, preference round
  trip, MLX adapter conversion, ROCm smoke, multi-GPU smoke,
  end-to-end auto-scaffold cycle, tokenized-cache unit suite, peer
  round-trip, directive fixture tree → finite adapter.

### Thanks

Five phases worth of work. Six audits caught real bugs, and the sway
submodule's differential tests produced the empirical floor that the
engine is behaviorally sound.

## [0.9.0] — target

First tagged release. Ships via the
[tenseleyFlow/homebrew-tap](https://github.com/tenseleyFlow/homebrew-tap)
(`brew tap tenseleyFlow/tap && brew install dlm`). Below v1.0 on
purpose — a human still needs to train + export + `ollama run` a real
document end-to-end before we claim the stable number.

### Highlights

- CLI: `init`, `train`, `prompt`, `export`, `pack`,
  `unpack`, `doctor`, `show`, `migrate`.
- Content-addressed store at `~/.dlm/store/<dlm_id>/` with atomic
  manifest updates and exclusive locking.
- Hardware-aware training plan (`dlm doctor`) across CUDA / MPS /
  ROCm / CPU tiers, with a refusal matrix that fails loudly on
  unsupported combinations.
- Curated base-model registry (10 entries) plus `hf:org/name`
  escape hatch with compatibility probes.
- LoRA + QLoRA training, replay-corpus retraining that retains prior
  sections, two-phase atomic version commits.
- Eval harness: val-loss, perplexity, early-stop.
- GGUF export with imatrix-calibrated quantization, explicit Go
  chat template (no fuzzy matching), embedding-row SHA verification,
  merge-safety gate against QLoRA pitfalls.
- Ollama integration: Modelfile emission, `ollama create`, smoke
  validation, closed-loop token-identity verification against the
  HF Jinja reference.
- `.dlm.pack` format: byte-identical packs, symlink / tar-bomb /
  zstd-bomb defenses, per-file SHA-256 integrity, pack-format
  migrations registry.
- Reproducibility contract: per-store `dlm.lock` with severity-table
  mismatch policy, `--strict-lock` / `--update-lock` / `--ignore-lock`
  CLI flags, determinism golden integration test.
- Documentation: getting started, `.dlm` format reference, CLI
  reference, six cookbook recipes, architecture overview,
  troubleshooting, determinism guide.
- Five starter templates: coding tutor, domain KB, writing partner,
  personal assistant, changelog.
- Weekly CI jobs: chat-template drift, slow integration suite.
- Pre-commit config: ruff, mypy `--strict`, non-slow pytest.

### Thanks

Built by following `.docs/findings.md` and the 29-sprint plan closely.
Every pitfall in the findings inventory corresponds to a test and an
explicit guardrail somewhere in the codebase.

---

The complete per-sprint history lives in `.docs/sprints/` (local to
the repo by user choice; planning artifacts stay out of git).
