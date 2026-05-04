# Audit 13 — Brutal end-to-end audit of the broader DLM promise

**Scope:** the marketing claim from `CLAUDE.md` — *"a text file with a `.dlm`
extension becomes a local, reproducible, trainable LLM. Edit the document,
retrain, share. Not a toy — LoRA/QLoRA on a real pretrained base, exportable
to Ollama."* Verified by running, not reading.

**Date:** 2026-04-29.
**Hardware:** Apple Silicon (Darwin 25.4.0 arm64, 18 cores, 48 GB unified
memory, MPS, no CUDA, no bitsandbytes, no FlashAttention). Doctor reports
`backend=mps`, `determinism_class=best-effort`.
**Wall-time actual:** ~33 minutes of in-band runs (23:36 → ~00:09 EDT)
within the 90-min budget. Heaviest single op: 11.4-min fortran train at
120 steps.
**Method:** B12.1 reproduction, full E2E-A through E2E-E from the prompt,
with sway as the measurement instrument.
**Artifacts:** `docs/audits/13-findings/{logs,fortran-fine-tune,sway-runs,exports}/`.

## Verdict

**YELLOW.** The promise is **mostly delivered, with named caveats**:

- **Trainable** — yes. Real `train` against SmolLM2-135M-Instruct on a
  2,021-file fortran corpus completed in 11.4 min on MPS, monotonic loss
  drop 1.96 → 1.69, eval_loss 1.83. Adapter, training_state.pt with
  RNGs, pinned_versions.json, metrics.sqlite all written.
- **Exportable to Ollama** — yes, end-to-end. GGUF base + adapter produced
  in ~20 s, registered with `ollama create`, smoke prompt produced
  coherent text. **But** `--no-template` is silently ignored at emission
  (M13.1).
- **Reproducible** — **partially**. Sway re-run is bit-exact deterministic;
  `dlm train --fresh --seed 42` re-run on the same tiny doc yields
  **different adapter SHAs** (`1afc...` vs `bb1f...`) on MPS.
  `training_state.rng.json` is bit-identical across runs, so the drift
  is in MPS-kernel ops, not RNG state. Doctor says
  `determinism_class: best-effort` — the docs are honest, but
  CLAUDE.md's "reproducible" headline is aspirational on Apple Silicon.
- **"Not a toy"** — **partially**. The 135M adapter measurably shifted
  the model on fortran-keyword prompts (sway delta_kl `+44.18σ` vs null
  baseline) but **did not internalize** the hand-written Q/A pairs
  (sway paraphrase_invariance `-3.51σ`, FAIL). Confirmed at runtime: the
  Ollama smoke output for "When should I use do concurrent?" produced a
  Python code block, not Fortran. The base 135M floor is too thin for
  the Q/A semantic-binding promise on a one-pass, 120-step run.
  Distribution-shift signal is real; transfer to question-answering is
  not.
- **Share** — yes. `dlm pack` → `dlm verify` → `DLM_HOME=… dlm unpack`
  → `dlm prompt` round-tripped a 15 MB bundle cleanly. **But**
  CLAUDE.md documents `dlm unpack PATH --home DIR`; the actual flag is
  `DLM_HOME=…` env-var only (M13.2).
- **"Edit the document, retrain"** — yes mechanically, but at ~3.5 s/step
  on MPS with 60-step eval pauses adding ~30s, the iteration loop is
  6-12 min for a 120-step run. Workable, not delightful.

The strongest evidence in this audit is **sway**: with the null_adapter
calibration probe in the suite, sway gives 4-decimal scores, z-scores,
and a single-line verdict that resolves "did training do anything?"
honestly. Sway re-runs are bit-identical. The audit is much stronger
as a result; without sway it would have been "the smoke output looked
fortran-ish" hand-waving.

The B12.1 blocker from Audit 12 is **fixed**. No new blockers found.

## What's fixed since Audit 12

- **B12.1 manifest race resolved.** Same fortran corpus + same source
  directives + fresh ULID → trained to completion. The audit-12 store
  (`01KQAR00VP2KAKVGCB7W4XRRZY`) now has a `manifest.json` and the
  failing path no longer reproduces.
- **M12.1 cli/commands.py extracted.** The 4,650-LOC monolith from
  Audit 12 is gone; `src/dlm/cli/commands/` is now a 23-file package,
  largest file 698 LOC (`train.py`), total 4,449 LOC. Restructured.

## Blocker findings

**None on the broader promise.** Every CLAUDE.md headline claim has a
working code path on a fresh box. Reservations are filed below as
majors and minors.

## Major findings

### M13.1 — `dlm export --no-template` is a no-op at Modelfile emission

**Live caught.** `dlm export … --no-template` suppresses the preflight
chat-template check, but the Ollama Modelfile's `TEMPLATE """…"""`
block is still written.

- Help text (`src/dlm/cli/commands/export.py:41`): "Skip writing
  TEMPLATE into the Modelfile."
- Plan path: `cli_no_template` is forwarded into
  `ExportPlan.include_template = not cli_no_template`
  (`src/dlm/export/plan.py:141`).
- **Only one consumer** of `plan.include_template` exists in `src/`:
  `src/dlm/export/runner.py:223`, where it gates the preflight check.
- `src/dlm/export/ollama/modelfile.py:101,124` calls
  `_build_template_block(template_row)` and appends it to the Modelfile
  unconditionally — `plan.include_template` is never consulted by the
  emission code.
- **Evidence** (`docs/audits/13-findings/logs/export-2-no-template.log`
  + the resulting Modelfile in
  `~/.dlm/store/01KQBN26S8RY8MGHE9HM09E8CM/exports/Q4_K_M/Modelfile`):
  ```
  FROM ./base.Q4_K_M.gguf
  ADAPTER ./adapter.gguf

  TEMPLATE """{{- if .System }}<|im_start|>system
  …
  ```
  The block is present even though `--no-template` was the only
  template-affecting flag.

**Severity:** MAJOR — the help text lies. Users counting on this flag to
test "what does Ollama do without a TEMPLATE override?" get the wrong
answer.

**Fix shape:** in `_build_template_block` consumers, check
`ctx.plan.include_template` and emit nothing (or a comment) when False.

### M13.2 — `dlm unpack --home DIR` is documented but not implemented

CLAUDE.md "CLI surface by release" lists:
```
dlm unpack <path> [--home DIR] [--force]
```

`uv run dlm unpack --help` shows:
```
--force   Overwrite an existing store with the same dlm_id.
--out     Directory to place the restored .dlm (default: alongside the pack).
```

Passing `--home DIR` aborts with `No such option: --home`. The actual
override is via env var (`DLM_HOME=… dlm unpack ...` works). Either:

- Add a `--home` flag matching the documented surface, OR
- Update CLAUDE.md to drop `--home` (and any other commands documented
  with it). The same pattern almost certainly applies elsewhere — a
  doc/CLI sync sweep is the right scope.

**Severity:** MAJOR for documentation-versus-binary mismatch on a
top-line CLI claim.

### M13.3 — `dlm metrics PATH --run-id 1` parse breaks (option-after-positional)

```
$ dlm metrics PATH --run-id 1
… error: Missing argument 'PATH'.
$ dlm metrics --run-id 1 PATH
run_id=1  phase=sft  seed=42  status=ok  steps=12  evals=4
```

Typer's group dispatch ate the positional. Workaround is "options
before positional," but every CLI user expects the opposite. Either:

- Restructure as a subcommand-group: `dlm metrics show PATH --run-id 1`
- Or pass `path` to the group's callback so subcommand parsing sees a
  consumed positional first.

**Severity:** MAJOR for UX paper-cut on a documented v2 surface
command. Trivially fixable.

### M13.4 — "Reproducible" is aspirational on MPS

`dlm train --fresh --seed 42 --max-steps 8` on a tiny 3-Q/A doc, run
twice in succession on the same hardware, in the same shell:

| Run | adapter_model.safetensors SHA |
|---|---|
| v0001 | `1afcd3f524e62dd17b87bf7059f698ab87882986e0397a353d5e7f3b358837e0` |
| v0002 | `bb1f67dbc19b7ebdba1910477f8d5cb23d4e5442092fada06f5d2817a3a137d6` |

Different bits. `training_state.rng.json` was bit-identical between
runs, so RNG-seed plumbing is correct — the drift is downstream of
RNGs, in MPS kernel ops. `dlm doctor --json` already reports
`determinism_class: best-effort` and the per-store `dlm.lock`
acknowledges this. So the implementation doesn't lie; **CLAUDE.md
does**, by stating "Edit the document, retrain, share" and labeling
the project "reproducible" without a platform caveat. On Linux+CUDA
with `CUBLAS_WORKSPACE_CONFIG=:4096:8` this is presumably bit-exact
(not verified in this audit); on macOS+MPS it isn't.

Also captured (m13.5 below): `--fresh` does **not** wipe and reset the
adapter version counter. v0001 from run 1 stayed; run 2 wrote v0002.
Tangential to determinism but related — interpreting `--fresh` as
"start from base weights for *this* run" is sensible, but a user might
reasonably expect "throw away prior versions and start over."

### M13.5 — `dlm show` re-expands directives on every invocation, on stderr

```
$ dlm show fortran.dlm --json 2>/dev/null | head
{ … clean JSON … }
$ dlm show fortran.dlm --json 2>&1 >/dev/null | wc -l
243
```

For a 2,021-file source-directive document, `dlm show --json` re-walks
the corpus and prints 243 lines of `dlm.directives.expand INFO:
directive: … exceeds max_bytes_per_file=32768; skipping` to stderr,
**every time**. `dlm show` is a read-only command users will run
repeatedly. The expansion result *should* be cached, and the INFO logs
should be silenced unless `--verbose`.

Mostly cosmetic until you script `dlm show` in a loop, at which point
it becomes a 1-2 second hit per call.

## Minor / informational findings

- **m13.6 — Stale brew install masks current binary.** `which dlm` →
  `/opt/homebrew/bin/dlm` → `dlm 0.9.0`, missing `repl, metrics,
  templates, push, pull, serve, verify, preference, synth, cache`
  (every command added since 0.9.0). The repo HEAD is `0.10.0` with
  the full surface. The release script either didn't bump the brew
  formula, or the user hasn't `brew upgrade`d. Symptom: a fresh
  contributor pasting commands from CLAUDE.md against the brew binary
  hits "No such command 'repl'." Either bump the brew formula on every
  `pyproject.toml` version bump (CI gate), or document `uv run dlm` as
  the canonical entry point in the repo's README.
- **m13.7 — ANSI escape spam from Ollama on non-TTY pipe.** Piping
  `ollama run … | tee` produces output salted with `[?25l[?2026h`
  cursor-control sequences (see `logs/ollama-fortran-prompt.log`). Not
  a dlm bug — Ollama 0.20.7 doesn't suppress its TTY UI under a pipe.
  Listed because it surfaces as garbled-looking dlm-export smoke
  output if anyone scripts the round-trip and doesn't `sed` it out.
- **m13.8 — Sway's `section_internalization` and `leakage` probes need
  the dlm bridge to be useful.** Both probes opt-out / SKIP without
  `ctx.sections` populated by the dlm-sway bridge. The bridge requires
  `pip install 'dlm-sway[dlm]'` plus the local DLM checkout, which I
  intentionally didn't install in the audit's sway venv (no clean way
  to do so without giving sway a typosquat-friendly PyPI install
  permission). On a fresh user's machine the bridge install is one
  pip line; it's worth shipping a starter spec that activates the
  bridge so users see attribution probes light up first time.
- **m13.9 — `dlm pack` produces unsigned bundles by default.** `dlm
  verify` correctly reports "is unsigned — no provenance.json
  inside." Signing is opt-in. Reasonable default; flagged because the
  Phase 7 promise of "share" is partly trust-signed sharing. A user
  expecting "I can verify this came from someone trustworthy"
  mid-distribution will discover the answer is "no, unless they
  signed it" only after running verify.
- **m13.10 — `--fresh` keeps prior adapter versions.** Two consecutive
  `dlm train --fresh ...` runs on the same store produced v0001 and
  v0002 side-by-side. If `--fresh` semantics are "fresh weights, but
  keep history," document it. If they're "wipe and start over," fix
  the version-counter reset.

## What works (earned praise)

- **Source-directive expansion at scale**: 2,021 .f90/.fypp files
  across 2 sources expanded in ~1.5 s, with sensible
  `max_bytes_per_file` skipping (62 oversized + 1 non-UTF-8) logged
  per-file. Tokenization cache populated 2,030 entries (3.1 MB).
- **Manifest contract honored.** After completion, the store has
  `manifest.json`, `adapter/versions/v0001/{adapter_config.json,
  adapter_model.safetensors, training_state.pt, training_state.rng.json,
  pinned_versions.json}`, `metrics.sqlite`, `replay/`,
  `tokenized-cache/`. CLAUDE.md pitfall #2 (training_state sidecar)
  observably honored.
- **Strict-mode parsing.** `dlm show` and `dlm train` on a malformed
  `.dlm` (bad ULID, wrong type, unknown key) return a single
  composed Pydantic error with line numbers and a one-shot diagnosis:
  ```
  error: /tmp/audit-e2e-e/malformed.dlm:2: dlm_id: Value error, dlm_id
  must be a 26-char Crockford base32 ULID, got 'not-a-ulid-just-a-string';
  training.lora_r: Input should be a valid integer, unable to parse
  string as an integer; training.unknown_key: Extra inputs are not
  permitted
  ```
- **Export pipeline produces the documented artifacts.** Within ~20 s
  on MPS: `base.Q4_K_M.gguf` (105 MB), `adapter.gguf` (1.8 MB),
  `imatrix.gguf` (631 KB), `imatrix.meta.json`,
  `export_manifest.json` with sha256 + llama_cpp_tag (`b8816`).
- **Ollama integration end-to-end.** `dlm export … --name dlm-fortran-audit13`
  registered `dlm-fortran-audit13:latest` (107 MB). `ollama run` produced
  coherent (though not fortran-flavored) output. The Modelfile's
  `TEMPLATE` block is the registry-authored Go template, not a Jinja
  fuzz-match (CLAUDE.md pitfall #1).
- **`dlm pack` / `dlm unpack` round-trip.** 15 MB bundle, restored
  cleanly into a fresh `DLM_HOME`, `dlm prompt` ran against the
  restored store immediately.
- **Per-store metrics.sqlite is real.** Tables `runs, steps, evals,
  exports, tokenization, gate_events, preference_mining`. Step-loss
  curve queryable via plain SQL — see
  `logs/step-loss.txt` for the audit's own run.
- **Sway is sharper than the docs claim.** Wall 3.2 s on a 4-probe
  + null_adapter suite, perfectly deterministic across re-runs,
  composite verdict + per-category scores + per-probe z-scores. The
  audit's central evidence — *the adapter shifted distributions on
  fortran prompts but did not bind Q/A pairs* — is z-scored with
  CI95.

## Promise audit table

| CLAUDE.md headline claim | Verdict | Evidence pointer |
|---|---|---|
| "A text file with a .dlm extension becomes a local … LLM" | **PASS** | `~/.dlm/store/01KQBN26S8RY8MGHE9HM09E8CM/{adapter,manifest.json,exports/Q4_K_M/}` after a 6.8 KB `.dlm` |
| "trainable LLM" (LoRA) | **PASS** | `logs/train-1.log` — 120 steps, train_loss 1.771, eval_loss 1.826; `metrics.sqlite` has the curve |
| "trainable LLM" (QLoRA) | **UNVERIFIED** | bitsandbytes is not available on Apple Silicon; no QLoRA path attempted |
| "real pretrained base" | **PASS** | `adapter_config.json:base_model_name_or_path = "HuggingFaceTB/SmolLM2-135M-Instruct"`; revision pinned in manifest |
| "exportable to Ollama" | **PASS** | `ollama list` shows `dlm-fortran-audit13:latest`; `ollama run` returned coherent text |
| "reproducible" | **PARTIAL** | sway: bit-exact reruns. dlm train on MPS: SHA drift between v0001 and v0002 with same seed (`logs/det-12-shas.txt`). Doctor honestly reports `best-effort` |
| "Edit the document, retrain" (workflow) | **PASS-with-caveat** | 11.4-min wall for 120 steps. `--watch` flag exists; not exercised this audit |
| "share" (pack/unpack) | **PASS** | `logs/pack.log` + `logs/unpack-2.log` + `logs/unpack-prompt.log` round-trip |
| "share" (push/pull/serve to HF/HTTP/LAN) | **UNVERIFIED** | not exercised this audit |
| "Not a toy" — the 135M model demonstrably learns a domain | **PARTIAL/FAIL** | sway: `dk_fortran` z=+44.18σ (distribution shift, real); `para_fortran` z=-3.51σ (Q/A binding **failed**). Adapter shifted token distributions on fortran-keyword prompts but did not internalize Q/A semantics in 120 steps. Visible at runtime: ollama smoke produced Python, not Fortran |
| "Pad token must NOT default to EOS" (CLAUDE.md pitfall #4) | **PASS** | tokenizer log says `pad_token_id: 0`, distinct from EOS |
| "training_state.pt sidecar" (pitfall #2) | **PASS** | `versions/v0001/training_state.pt` (7.5 MB) + `training_state.pt.sha256` + `training_state.rng.json` (16 KB) |
| "merge_and_unload on QLoRA refused without --dequantize" (pitfall #3) | **PASS-by-code-read** | `src/dlm/export/plan.py:102` and `src/dlm/export/merge.py` enforce; not live-tested (no QLoRA on MPS) |
| `dlm doctor --json` reports plan + capabilities | **PASS** | `logs/doctor-uv.json` |
| All v1.0 CLI commands present | **PASS** (binary `0.10.0`) / **FAIL** (brew `0.9.0`) | `dlm 0.10.0 --help` lists all v1.0 + v2 commands; `/opt/homebrew/bin/dlm` is 0.9.0 and missing `repl, metrics, templates, push, pull, serve, verify, preference, synth, cache` |
| `dlm unpack --home DIR` documented flag | **FAIL** | `--home` is not a flag; `DLM_HOME=` env var is the actual override (M13.2) |
| `dlm export --no-template` does what it says | **FAIL** | preflight is suppressed, emission is not; M13.1 |

## Methodology notes

Order of operations:

1. Read `docs/audits/12-brutal-post-audit-11.md`. Confirmed B12.1 is
   the open blocker to verify.
2. `dlm doctor --json` and `dlm --help` via both `which dlm`
   (`/opt/homebrew/bin/dlm` 0.9.0) and `uv run dlm` (repo HEAD 0.10.0).
   Discovered the brew/repo skew immediately — every subsequent
   command used `uv run dlm`.
3. `dlm init` to scaffold a fresh `.dlm`, then overwrote with the
   audit-12 fortran source-directive frontmatter, refreshed to schema
   v15 and a fresh ULID. Two source directives:
   `~/GithubOrgs/FortranGoingOnForty` (1,847 .f90/.F90/.f95) and
   `/tmp/stdlib_build/src` (174 .f90/.fypp). Total post-skip: 2,021
   files, 7.97 MB content.
4. `dlm train --fresh --seed 42 --max-steps 120`. **B12.1 did not
   reproduce** — manifest written, training proceeded. 11.4 min wall.
   Loss monotonic 1.927 → 1.886 → 1.805 → 1.689 → 1.638 → 1.607
   over six log points. Eval at steps 60/90/120; final
   eval_loss=1.826.
5. Captured adapter SHA snapshot. Authored `sway.yaml` with
   `delta_kl + paraphrase_invariance + calibration_drift + leakage`,
   first-pass schema mismatch on `section_internalization` (audit-author
   error), reauthored, ran. First sway run had no calibration → re-ran
   with `null_adapter` probe added.
6. `dlm export ... --quant Q4_K_M` (with `--no-template` after
   preflight refused without it — see M13.1). Inspected emitted
   artifacts. `--name dlm-fortran-audit13` registered into Ollama.
   `ollama run` smoke prompt.
7. Determinism: tiny 3-Q/A doc with a fresh ULID, `dlm train --fresh
   --seed 42 --max-steps 8` twice. Compared
   `adapter_model.safetensors` SHAs and
   `training_state.rng.json` byte-diff. Re-ran sway against the
   spec twice; compared per-probe `(verdict, score, raw, z)` tuples.
8. Failure-mode hunting: `dlm prompt` on never-trained doc; `dlm
   train` on malformed frontmatter; `dlm export --merged` on plain
   LoRA; 200 KB inflated `.dlm` parses; `dlm metrics PATH --run-id
   1` (broke); `dlm unpack --home` (missing flag).
9. `dlm pack` / `dlm verify` / `dlm unpack` (with `DLM_HOME=`) round-
   trip and prompt-on-restored-store.

Cleanup: `ollama rm dlm-fortran-audit13`.

The single biggest methodology learning: **if the audit had skipped
the `null_adapter` probe in the sway suite, every probe would have
shown verdict but no z-score, and the "did training do anything?"
question would have been a vibes call**. Sway with calibration is
properly falsifiable; sway without it is just a per-probe score
table. The dlm cookbook's starter sway specs should default-include
`null_adapter`.

## Summary in one paragraph

The promise is real for "a `.dlm` file becomes a trainable, exportable,
shareable LoRA on top of a real HF base." The 11.4-min train, the
working Ollama round-trip, the clean pack/unpack, the strict Pydantic
parsing, and sway's z-scored verdict on a real corpus all back this
up. The promise is **overstated** in two places: "reproducible" needs
a platform caveat (best-effort on MPS, verified non-bit-exact even
with the same seed), and "Not a toy" understates how much the 135M
floor needs help — distribution shift is real, Q/A internalization
is not, on a single 120-step pass. The two help-text bugs (M13.1
`--no-template` no-op, M13.2 `--unpack --home` missing) and the doc
drift (`/opt/homebrew/bin/dlm` is 0.9.0) are minor in code but
material in trust: a careful user reading CLAUDE.md and running the
brew binary will hit "No such command 'repl'." within 30 seconds.
None of this is a blocker; all of it is a documentation/UX sweep.
