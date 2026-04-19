# DocumentLanguageModel — Codex Session Boot Context

> This file is read on every session start. Keep it dense, authoritative, and
> aligned with the living docs. When anything here conflicts with `.docs/`,
> `.docs/` wins and this file must be updated.

## One-line

A text file with a `.dlm` extension becomes a local, reproducible, trainable
LLM. Edit the document, retrain, share. Not a toy — LoRA/QLoRA on a real
pretrained base, exportable to Ollama.

## Current stage

- ✅ Stage 1 — Planning & reference exploration (see `.docs/findings.md`)
- ✅ Stage 2 — Revised overview + 29 sprint files across 7 phases
- ✅ Stage 3 — this file
- ✅ Stage 4 — Audit 01 (YELLOW → patched). Blockers F01–F04 and majors
  F05–F22 triaged into Sprint 12b + inline sprint amendments. See
  `.docs/audits/01-initial-plan-audit.md` and the end of this file for
  the triage summary.
- ⏳ Stage 5 — Implementation (begin at Sprint 01)

## Where things live

```
.docs/overview.md             Canonical project description (read this first)
.docs/findings.md             Stage 1 digest from 8 parallel ref explorations
.docs/sprints/00-index.md     Master index of the 29 sprints
.docs/sprints/phase-*/        Sprint files; each has DoD and risks
.docs/audits/                 Stage 4+ audit outputs
.refs/                        Cloned reference repos (gitignored)
AGENTS.md                     You are here. Gitignored.
```

`.docs/` and `AGENTS.md` are in `.gitignore` by user choice — planning
artifacts stay local.

## Crystallized architecture

**Training paradigm**: LoRA / QLoRA on a user-selected pretrained base. No
from-scratch transformers. The base registry ships with Qwen 2.5
(0.5B–3B + Coder-1.5B), Llama-3.2 (1B, 3B), SmolLM2 (135M–1.7B), and
Phi-3.5-mini. Any HF model via `hf:org/name` with compatibility probes.

**Document shape**: `mydoc.dlm` is a single UTF-8 text file — YAML
frontmatter + markdown body with section fences (`::instruction::`,
`::preference::`, default-prose). A stable `dlm_id` in the frontmatter
binds the document to a content-addressed store at `~/.dlm/store/<dlm_id>/`.

**Retention**: single rolling adapter trained on the current document +
recency-weighted sample from a zstd-compressed replay corpus accumulating
every prior document version. Rejected alternative: versioned adapters
with weighted merge (LoRA-only, SVD cost, harder determinism).

**Export**: separate `base.gguf` + `adapter.gguf` + generated Modelfile with
`ADAPTER` directive. `--merged` opt-in produces a single file (QLoRA
requires explicit `--dequantize`).

**Hardware tiers**:
- NVIDIA CUDA (SM ≥ 8.0): first-class, bf16 + QLoRA 4-bit + FlashAttention
- NVIDIA CUDA (SM < 8.0): second-class, fp16 LoRA
- Apple Silicon MPS: first-class training (fp16 LoRA), optional MLX inference in Phase 5
- CPU: inference-only by default, training refused except `--force` on ≤200M bases
- AMD ROCm: experimental; Phase 5 promotes to Tier 2

## Stack

**In**: Python 3.11+, PyTorch ≥ 2.4, HuggingFace `transformers`/`peft`/`trl`/
`accelerate`/`datasets`, `bitsandbytes` (CUDA-gated), `safetensors`,
`zstandard`, llama.cpp (vendored git submodule) for GGUF export,
Ollama (user-installed), `typer`, `rich`, `uv`, `pytest`, `mypy --strict`,
`ruff`.

**Out**:
- Unsloth (monkeypatch fragility, transformers-version pinning hell, CUDA-only, Apple Silicon excluded)
- MLX for training (adapter `.npz` format is not PEFT-compatible)
- From-scratch transformers
- DeepSpeed / ZeRO through v1.0
- Windows first-class (best-effort; Linux + macOS are supported tiers)

## Pitfalls to always remember

1. **Ollama uses Go `text/template`, not Jinja2.** The GGUF's Jinja
   chat-template is fuzzy-matched by Ollama and fails silently when
   unmatched. We always emit an explicit `TEMPLATE "..."` in the Modelfile
   from our per-base-model Go template registry. Round-trip tests assert
   token-identity with the HF Jinja reference.

2. **`peft.save_pretrained` does NOT save optimizer / scheduler / RNG.** We
   write a separate `training_state.pt` sidecar with optimizer state,
   scheduler state, AMP scaler, torch/cuda/numpy/python RNGs, step, epoch,
   pinned versions. Without this, resume is not deterministic.

3. **`merge_and_unload` on 4-bit QLoRA base is precision-unsafe.** Refuse
   the merged export path on QLoRA unless `--dequantize` is explicit; then
   dequantize to fp16 before merge.

4. **Pad token must NOT default to EOS.** Label corruption when EOS
   appears mid-sequence. Fallback: unk_token → else add `<|pad|>` (and
   then `modules_to_save=["embed_tokens","lm_head"]` is forced, inflating
   adapter size; warn loudly).

5. **Pre-tokenizer hash table in llama.cpp** is a silent-failure surface.
   Sprint 06 probes at registry-build time + on `dlm init --base hf:...`;
   Sprint 11 re-verifies at `dlm export` preflight. Bumping
   `vendor/llama.cpp` re-runs the registry probe suite via
   `scripts/bump-llama-cpp.sh`.

6. **Sample packing without FlashAttention** causes `position_ids` drift on
   MPS. Doctor disables packing when FlashAttention is unavailable and
   packing is otherwise unsafe.

7. **`target_modules="all-linear"` on small models** causes memory blowup
   and instability. Use the per-architecture registry from sprint 06 as
   the default.

8. **Determinism is a contract**: fixed seed, `use_deterministic_algorithms`,
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`, pinned versions recorded in
   `dlm.lock`. Any code change that breaks the golden determinism test is
   a breaking change.

Full inventory in `.docs/findings.md#9`.

## Contract boundaries (audit F25)

Four load-bearing files; keep them distinct when editing.

- **`manifest.json`** (per-store): running narrative of training runs,
  exports, content hashes, adapter version. Mutable on every run. Owned
  by Sprint 04; extended by Sprints 09, 11, 12, 12b.
- **`dlm.lock`** (per-store): version pins + hardware tier + determinism
  flags + license acceptance fingerprint. Written once per run; stable.
  Owned by Sprint 15; extended by Sprint 12b (license) and Sprint 23
  (world_size + accelerate).
- **`training_state.pt`** (per-store, per-adapter-version): optimizer,
  scheduler, scaler, all RNGs, step/epoch. Required for bit-exact resume.
  Owned by Sprint 09. Two-phase commit with adapter directory.
- **`exports/<quant>/export_manifest.json`** (per-export): checksums,
  quant level, pinned llama.cpp tag, smoke output. Owned by Sprint 11;
  appended via Sprint 12.

And one repo-level file:

- **`dlm.lock`** at the repo root: records which `(torch, transformers,
  peft, trl, bnb, platform)` tuples have a checked-in determinism golden.
  Different from the per-store `dlm.lock`. Owned by Sprint 15.

## Development guidelines

- **Commit often, commit small.** Avoid monolithic commits; maximize commits
  per feature so the history shows a narrative. One commit per distinct
  change (a file, a config, a fix), not per day's work.
- **Commit message style**: imperative, terse, one line unless a technical
  choice requires elaboration. **No coauthorship** on any commit.
- **Avoid `git add -A`.** Stage specific files by name; it's harder to
  leak secrets or commit unrelated changes.
- **No shortcuts when a robust approach exists.** If you find yourself
  writing "the simplest approach is…", stop and ask whether this produces
  a trainable LLM. If not, reapproach.
- **Senior AI-engineering discipline.** Write efficient, well-engineered
  code. Respect the pitfall inventory.
- **Strict validation, fail fast.** Axolotl's permissive warnings are the
  anti-pattern. Our Pydantic schemas reject unknown keys, wrong types, and
  inconsistent combinations at parse time.
- **Determinism is a contract.** See above.
- **Tests before implementation** for anything touching training dynamics,
  tokenization, or GGUF export. The tiny-model fixture (sprint 02) makes
  end-to-end CI feasible; use it.
- **`mypy --strict` from day one.** Never loosen; fix the type at source.
- **Per-sprint definition of Done is binary.** A sprint is not Done until
  every DoD checkbox passes and the sprint file is marked Done.

## Workflow inside a sprint

1. Read `.docs/sprints/phase-N/NN-*.md` in full.
2. Cross-check against `.docs/findings.md` where the sprint references
   pitfalls or patterns (the sprints do cite sections).
3. Implement incrementally. Commit per file / per logical unit.
4. Write tests alongside (or before) the code.
5. Check every DoD item manually before flipping Status to Done.
6. Update `.docs/sprints/00-index.md` status column if we maintain one.

## CLI surface by release

**v1.0** (Phase 3 end):
```
dlm init <path> [--base <key>] [--template <name>] [--i-accept-license]
dlm train <path> [--resume|--fresh] [--seed N] [--max-steps N] [--gpus ...]
                 [--strict-lock|--update-lock|--ignore-lock]
dlm prompt <path> [query] [--max-tokens N] [--temp F] [--adapter <name,...>]
dlm export <path> [--quant Q] [--merged [--dequantize]] [--name N] [--no-smoke]
                  [--adapter-mix name:w,...]
dlm pack <path> [--out X] [--include-exports] [--include-base
                [--i-am-the-licensee <url>]]
dlm unpack <path> [--home DIR] [--force]
dlm migrate <path> [--dry-run] [--no-backup]
dlm doctor [--json]
dlm show <path> [--json]
```

**v2** (Phases 4–6):
```
dlm repl <path>
dlm train <path> --watch [--repl]
dlm metrics <path> [--json|--csv]
dlm metrics watch <path>
dlm templates list [--refresh]
```

**v2+** (Phase 7):
```
dlm push <path> [--to hf:org/name | --to <url>] [--sign]
dlm pull <source>
dlm serve <path> [--public [--i-know-this-is-public]]
```

## Stage gates

- Stage 4 — **Patched (YELLOW → triaged)**. New Sprint 12b owns F01–F04.
  17 majors amended inline into existing sprints. 9 minors deferred to
  first touch of their owning sprints. A re-audit pass is recommended
  before declaring GREEN and entering Stage 5.
- Stage 5 — begin Sprint 01 (scaffolding) once Stage 4 is GREEN.

## Context for future sessions

- Always load `.docs/overview.md`, `.docs/findings.md`, and
  `.docs/sprints/00-index.md` before working on a sprint. Skim the
  relevant sprint file in full.
- The user prefers concise, direct engineering discussion. Surface
  tradeoffs; make recommendations with reasoning.
- When in doubt about an implementation choice, check findings §10
  (adoption matrix per reference repo) — it's the opinionated source of
  truth for "why are we doing it this way, not that way."


<claude-mem-context>
# Memory Context

# [DocumentLanguageModel] recent context, 2026-04-19 7:40pm EDT

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (20,314t read) | 1,271,868t work | 98% savings

### Apr 18, 2026
92 5:26p 🔵 armfortas/fortsh Build Produces Widespread Ambiguous USE Import Warnings
94 5:27p 🔵 fortsh Full Build Succeeds via armfortas — Complete Object Link Map Confirmed
98 5:29p 🔵 fortsh Smoke Tests Pass — Parameter Expansion and Pipeline Basics Verified
99 " 🔵 fortsh Test Suite Results — read 94%, var-ops 80% with Identified Failure Clusters
100 " 🔴 Null Pointer Dereference in afs_compare_char When Empty String Variable Used in Parameter Expansion
101 5:32p 🔵 Empty-String Parameter Expansion Bug Isolated to Assignment Side-Effect, Not Expansion Engine
105 5:34p 🔵 V="" Assignment Alone Crashes via execute_ast_node — Bug Is in Assignment Executor, Not Compound Commands
111 5:36p 🔵 armfortas IR Builder Architecture — FuncBuilder API Surface Mapped
113 5:38p 🔵 SIGSEGV Confirmed — Dynamic Substring Index on Zero-Length Allocatable Char Crashes
117 5:39p 🔵 fortsh Crash Site Confirmed in ast_executor.f90 — Dynamic Substring on Zero-Length Allocatable
120 5:40p 🔴 lower_substring_full — Dynamic Substring Out-of-Bounds GEP Fixed with Safe Clamp
121 5:42p 🔴 substring fix validated — 8/8 substring tests pass, repro RC=0, fortsh build proceeding without errors
137 5:55p 🔵 armfortas allocate(scalar_derived) Skips Field Default Initializers
138 " 🔵 fortsh IFS / read Builtin Architecture Confirmed
139 5:56p 🔴 armfortas: allocate(scalar_derived) Now Applies Field Default Initializers
142 5:59p 🔵 fortsh Build Completes with Ambiguous USE Import Warnings in readline Module
143 6:01p 🔵 fortsh Build Produces Ambiguous USE Import Warnings from Duplicate Module Exports
145 " 🔵 armfortas trim/adjustl Branch Produces Correct Value but print '(a)' Adds Leading Space
146 6:03p 🔵 armfortas print '(a)' Emits Carriage-Control Space — Confirmed by od Byte Dump
147 " 🟣 Regression Test Added: allocatable_shell_default_ifs_follows_trim_branch
148 " 🔵 fortsh Builtin Test Results: read 100%, arithmetic 100%, variable_ops 85%, arrays 0% on literal init
149 " 🔵 fortsh Array Literal Init Bug — Bounds Check Failure: index 1 outside [1, 0]
153 6:06p 🔵 Array Section Argument Descriptor Bug — values(1:count) Passed as Assumed-Shape Gets upper=0
155 6:14p 🔵 armfortas Emits Duplicate Ambiguous-USE Warnings Per Translation Unit
156 " 🔵 fortsh Makefile Has Full Native armfortas Profile
157 " 🟣 armfortas Rust Test Suite — Array Section Bounds Test Passing
158 " 🔵 fortsh Binary Previously Built by armfortas — Incremental Rebuild in Progress
159 " 🔵 fortsh test_variables_simple Uses Pooled String API — Not Standard Fortran Variables
160 6:15p ✅ armfortas Array Section Fix Staged for Commit — lower.rs and cli_driver.rs
161 " 🔴 armfortas Commit 4ec3e9a — Lower Array Section Descriptor Actuals
162 6:16p 🔵 fortsh Incremental Rebuild with armfortas Completed Successfully
163 6:17p 🔵 armfortas Working Tree — Active afs-as/afs-ld Changes Plus Repro Test Artifacts
177 6:24p 🔵 fortsh Build — Mass Ambiguous USE Import Warnings from armfortas
179 6:27p 🔵 armfortas Peak RSS ~99 MB Compiling fortsh lexer.f90
182 6:30p 🟣 fortsh Binary Successfully Built with armfortas — /tmp/fortsh_armf_arrayfix/bin/fortsh
183 6:31p 🔵 fortsh 1.7.0 Binary Verified Functional — Basic Array and Pipeline Semantics Correct
184 " 🔵 Array Test Suite Baseline — 17/31 Pass (54%), 14 Failures Cataloged
192 6:34p 🔵 Test Harness Uses Bash 3.2 as Reference — Assoc Array Failures Are Baseline Artifacts
193 " 🔵 Three armfortas-Specific Array Regressions Confirmed Against flang-ref Baseline
220 6:50p 🔵 armfortas Unset Module Variable — Parity with flang-new Confirmed
222 " 🔵 fortsh Array Unset Bugs — Two Distinct Failures in armfortas Build vs Correct flang Reference
223 6:53p 🔵 fortsh Null-Assignment Hole (`arr[1]=`) Produces Correct Sparse Indices in armfortas Build
224 " 🔵 AST Executor Does Not Dispatch `unset` as Builtin — "command not found" at Runtime
228 " 🔵 armfortas `builtin_unset` Direct Call Crashes — "Bounds check failed: index 1026 outside [1, 1025]"
231 6:54p 🔵 armfortas `unset foo[N]` on Non-Existent Variable Crashes — Bug Not Array-Existence-Dependent
232 " 🔵 fortsh `execute_simple_command` Builtin Dispatch — Routes Through `execute_pipeline`, Not Direct Call
234 6:56p 🔵 armfortas `unset` Bug Scope Narrowed — Scalar `unset` Works, Array-Index Form Always Crashes
240 7:39p 🔵 armfortas expand_out Token Output Shows Null-Byte Corruption
241 " 🔵 flang-new Cannot Compile expand_out Repro — `fill` Not Found in Module `m`
242 7:40p 🔵 armfortas .amod Exports `fill` With Fixed-Length Allocatable Character(len=32) Intent(out)

Access 1272k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>