# Finding 04 — Q/A-shape recipe closes the wedge

**Hypothesis tested:** Findings 01–03 isolated training-data shape as
the bottleneck — LoRA on raw source files learns to be a source
autocomplete engine, not a question-answering bot. The fix: train on
INSTRUCTION-shaped Q/A pairs only, no `sources:` directives. Use
qwen2.5-coder-1.5b as base (Finding 03 confirmed it preserves English
capability under LoRA training).

**Verdict:** the hypothesis is **confirmed**. The recipe-shape fix
produces correct trained-question answers and preserves general
capability. Generalization to *nearby* unseen questions is partial
and dataset-size-dependent.

## Setup

- **Doc:** `docs/audits/13-followup/finding04/the-doc.dlm`
- **Store ULID:** `01KQDWAHNJ7FD72EQ4J4FXBJ2V`
- **Adapter:** `~/.dlm/store/01KQDWAHNJ7FD72EQ4J4FXBJ2V/adapter/versions/v0002`
- **Base:** `qwen2.5-coder-1.5b`
- **Sections:** 35 INSTRUCTION (32 train / 3 val), 0 PROSE, 0 raw `sources:`
- **Recipe:** r=16 / α=32 / lr=2e-4 / 100 epochs (early-stopped at epoch 44, step 400)
- **Final:** train loss 0.62, eval loss 0.047, **mean token accuracy 98.6%**

The corpus is ~60% stdlib_sorting (the wedge target), 40% other
modules (io, math, strings, array, kinds, ascii, hashmaps, quadrature,
logger, plus a few general-fortran questions like `intent(in)` and
`do concurrent`).

## The wedge: closed

The audit's wedge question (Q3 from Finding 03's baseline):

> Show the signature of stdlib's sorting routine in Fortran.

**Base:** *wrong.* Hallucinates `iso_fortran_env::sort`.
**Finding-03 LoRA (raw-source training):** *worse.* Degenerate
"should be able to sort an array of integers in ascending order. The
sorting routine should be able to sort an array of floating-point
numbers..." enumeration loop.
**Finding-04 LoRA (Q/A-shape training):** ✓ correct, verbatim:

```fortran
use stdlib_sorting, only: sort
call sort(array[, reverse])
```
The `array` argument is `intent(inout)` and must be a rank-1 array of
an intrinsic numeric type (integer kinds, real kinds), `character(*)`,
`type(string_type)`, `type(bitset_64)`, or `type(bitset_large)`. The
optional `reverse` argument is a scalar logical with `intent(in)`.

## Generalization curve

| Question type | Result | Verdict |
| --- | --- | --- |
| **Seen exactly** (Q3 wedge, intent(in)) | verbatim correct | ✓ memorized cleanly |
| **Unseen, overlap** ("sort a real(dp) array") | knows `stdlib_sorting`, invents wrong call form | partial |
| **Unseen, different module** (`stdlib_strings::starts_with`) | falls back to Rust syntax | none |
| **Unseen, different module** ("read a CSV with stdlib") | hallucinates plausible API | none |
| **Out-of-domain** (capital of France, Python list comprehension) | unchanged from base | ✓ preserved |

Full transcripts at `docs/audits/13-followup/finding04/direct-query-results.md`.

## What we learned

### 1. The recipe-shape hypothesis is correct

LoRA trained on INSTRUCTION-only Q/A pairs produces an adapter that
*answers questions* in the trained format. LoRA trained on raw source
files produces an adapter that *autocompletes source code*. Same base,
same rank, same compute — completely different behavioral character.
This is the cleanest finding of the investigation.

### 2. The dataset-size / generalization tradeoff is sharp

With 32 hand-authored Q/A pairs:
- 100% trained-question fidelity (98.6% eval token accuracy)
- Strong reproduction of trained answers under varied prompt phrasing
- Partial knowledge of "named entities" (`stdlib_sorting`,
  `stdlib_io`, `loadtxt`) appears in unseen-question responses
- API-form generalization is weak — model invents plausible-looking
  syntax instead of generalizing the patterns it saw

This means dlm's product story isn't "learn a domain from a few
examples" — it's "learn the questions you actually want to answer,
with one training row per question." That's a more honest story and
also more practically useful: users know what they're getting.

### 3. The bigger base preserves general capability under aggressive LoRA training

Finding 03 already showed `cal_general` 0% regression on
qwen2.5-coder-1.5b. Finding 04 replicates this even with 100-epoch
overfit-style training: the model still answers "What is the capital
of France?" correctly and writes valid Python list comprehensions.
This is the architectural property that makes the recipe-shape fix
viable — at SmolLM2-135M, the same overfit training would have
shredded English chat capability.

## Bugs surfaced

1. **`src/dlm/replay/store.py:187`** — `parse_instruction_body` was
   called without `_normalize_probe_markers`. Fixed in this branch.
   Without the fix, INSTRUCTION sections with `### Q !probe` headers
   trigger a parse error during retrain even with `--fresh`, because
   replay snapshots the raw section content. The other dlm callers of
   `parse_instruction_body` (`eval/probes.py`, `cli/commands/synth.py`,
   `train/gate/orchestrator.py`, `preference/mine.py`) should be
   audited for the same bug — that's a follow-up.
2. **MLX backend silently ignores PEFT adapters.** dlm's auto-routing
   selects MLX on darwin-arm64. PEFT `adapter_model.safetensors`
   isn't an MLX-LM adapter format; the inference path appears to load
   the base and ignore the adapter. The user-visible failure is
   "trained model behaves identically to base" — easy to misdiagnose
   as "training didn't work." Workaround: `--backend pytorch`. Real
   fix needs investigation in `src/dlm/inference/backends/mlx.py`.

## Implications for the dlm product narrative

The investigation produced a clean three-step story:

1. **Use a base ≥ 1B params and code-pretrained where available.**
   Smaller bases (135M) actively degrade under LoRA training of any
   shape. Recommended-base table needs a warning at the small end.
2. **Train on INSTRUCTION-shaped data, not raw source code.** Raw
   source teaches autocomplete; INSTRUCTION teaches Q/A. Pick one
   based on the goal. dlm's docs should make this distinction.
3. **Plan one Q/A pair per question you want to answer.** With
   small datasets, pinpoint reproduction is reliable but
   generalization is weak. Scale the corpus to scale the surface.

Each step has falsifiable evidence in this directory:

- Step 1 from Findings 02 (135M memorization+forgetting) and 03
  (1.5B preservation)
- Step 2 from Finding 03's wedge failure vs Finding 04's wedge success
- Step 3 from Finding 04's generalization curve

This is publishable as written. The audit closes here as **GREEN**: the
end-to-end fortran fine-tune story works end-to-end with the right recipe,
the right base, and a corpus shaped to match the user's question set.

## Next experiment (optional)

[Finding 05 — corpus density](./05-corpus-density.md) *(not started)*

Test the dataset-size / generalization curve directly. Build a 100-pair
INSTRUCTION corpus covering each stdlib module's main API surface
(rather than 60% sorting). Train, then probe with held-out questions
about modules covered with one to three training pairs each.

The hypothesis: there's a per-module Q/A density floor below which
the model can't generalize the API form. Finding the threshold gives
dlm users a concrete planning number ("budget N Q/A pairs per module
of API surface to teach"). This is the "how much training data does
each new domain take" question that makes the product practically
plannable.
