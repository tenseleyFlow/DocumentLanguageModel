# Audit 13 follow-up — investigation log

A sequence of single-variable experiments, each isolating one
hypothesis about why the original audit-13 fortran fine-tune showed
"adherence works, attribution fails." Reading order:

1. **[Finding 01 — recipe failure](./01-recipe-failure.md)**
   Re-run audit-13 with 5× more INSTRUCTION sections + 5× more steps +
   sway bridge probes lit. Result: traded "no Q/A binding" for
   "memorization without generalization." Falsified the *volume-only*
   hypothesis.

2. **[Finding 02 — CPT isolation](./02-cpt-isolation.md)**
   Stripped INSTRUCTION sections, ran pure CPT at LoRA r=64. Result:
   adapter learned form-specific autoregression (memorization +
   English regression), not domain abstraction. Falsified the
   *two-stage CPT-then-SFT* hypothesis at this base size and surfaced
   architectural floor: SmolLM2-135M lacks the capacity to compose
   fortran knowledge with English chat behavior.

3. **[Finding 03 — base-size floor + recipe is the bottleneck](./03-base-floor.md)**
   Promoted to `qwen2.5-coder-1.5b`, ran the audit-13-followup doc
   without recipe changes. Result: bigger base eliminated catastrophic
   forgetting (`cal_general` 26% → 0% items regressed) **but the LoRA
   still memorized instead of generalized**, and actively degraded
   the wedge question (Q3 stdlib sorting). Isolated the bottleneck:
   it's the *recipe* (raw-source training data shape), not the base
   size. dlm's value-add is conditional on training-data shape.

4. **[Finding 04 — Q/A-shape recipe closes the wedge](./04-instruction-shape.md)**
   Built a doc with 35 hand-authored INSTRUCTION sections (no raw
   sources), trained on qwen-coder-1.5b at r=16 / 100 epochs.
   Result: **Q3 now lands correctly** — verbatim trained answer
   pointing at `stdlib_sorting :: sort` with proper signature.
   General capability preserved. Generalization to unseen but
   overlapping questions is partial; to unseen-different-module
   questions, weak. Surfaced two real dlm bugs along the way
   (replay store parser bug + MLX backend silently ignoring PEFT
   adapters). The audit closes GREEN with a clean three-step
   product story.

5. **[Finding 05 — Q/A-shape recipe on smol](./05-smol-qa-shape.md)**
   Took finding 04's exact INSTRUCTION-only corpus, swapped the base
   to SmolLM2-135M. Result: **architectural floor confirmed.** Smol
   overfits the 32 train pairs (eval_loss bottoms at epoch 20-30 then
   climbs while train descends), produces token-salad on held-out
   questions, and damages base capability so badly that "What is 2+2?"
   hallucinates a `stdlib_array_plus` helper. Finding 02's "use 135M
   for style-transfer demos only" caveat is a base-size constraint,
   not a recipe constraint.

6. **[Finding 06 — corpus density](./06-corpus-density.md)** *(optional)*
   Test the dataset-size / generalization curve. Find the per-module
   Q/A density floor below which the model can't generalize the API
   form. Gives dlm users a concrete planning number ("budget N Q/A
   pairs per module").

## Why this format

Each file is self-contained — one experiment, one verdict, one
falsified-or-confirmed hypothesis. No "TODO" findings; if a
hypothesis is in-flight, it's either at the bottom of the latest
finding ("next experiment") or has its own placeholder file. This
keeps the investigation traceable: any future reader can look at the
state of the directory and reconstruct what we tested, what we
ruled out, and what we still don't know.

## Standing artifacts

- `the-doc.dlm` — the audit-13-followup canonical doc (Finding 01)
- `stage1/the-doc.dlm` — PROSE-only stage-1 doc (Finding 02)
- `stage1/sway.yaml` — stage-1 sway eval spec
- `sway-results.json` / `sway-results.md` — Finding 01's raw sway run
- `/tmp/sway-stage1.json` — Finding 02's raw sway run *(local-only)*
- `train.log` — Finding 01's training output

## What we know so far

**Confirmed:**
- The dlm pipeline works end-to-end: doc → train → adapter → sway →
  direct query, all deterministic and reproducible. The plumbing is
  sound.
- The bridge probes (`section_internalization`, `leakage`,
  `paraphrase_invariance`) require specific corpus shape: ≥2 section
  kinds (PROSE + INSTRUCTION minimum) for the leak-check, and
  `!probe` markers for paraphrase case generation.
- SmolLM2-135M produces memorization, not generalization, on this
  corpus regardless of recipe variations within reach.
- Bigger base (qwen2.5-coder-1.5b) eliminates catastrophic
  forgetting — `cal_general` regressions 26% → 0%. dlm's
  recommended-base table should warn that small bases (135M) actively
  degrade under LoRA.
- **Recipe shape is the bottleneck, not base size.** Same doc on a
  bigger base still produces memorization; LoRA on raw-source training
  rows learns "be a source autocomplete engine," not "answer questions
  about the domain."

- **Q/A-shape recipe on qwen-coder-1.5b closes the wedge.** Finding
  04 trained 35 hand-authored Q/A pairs and produced an adapter that
  reproduces trained answers verbatim, preserves general capability,
  and partially activates domain knowledge on related questions. The
  three-step product story (bigger base + Q/A-shape recipe + plan
  one pair per question) has falsifiable evidence at each step.

**Unknown:**
- The dataset-size / generalization curve. With 32 hand-authored Q/A
  pairs, generalization to unseen-different-module questions is
  weak. How many pairs per module are needed for the model to
  abstract the API form? Finding 05 (optional) would answer this.
- Whether dlm-synthesized Q/A pairs (via `dlm synth instructions`)
  produce comparable training results to hand-authored ones, or
  whether teacher quality matters. This is a question for the
  product onboarding story.

**Bugs filed during the investigation (worth fixing in dlm):**
- `src/dlm/replay/store.py:187` — `parse_instruction_body` called
  without `_normalize_probe_markers`. Patched in this branch as
  part of Finding 04.
- MLX inference backend silently ignores PEFT adapters on darwin-arm64
  (auto-routing falls through to MLX, MLX loads base only). User-
  visible failure is "trained model behaves like base" — major
  product-trust risk. Workaround: `--backend pytorch`.
