# Finding 03 — base-size floor + the recipe is the bottleneck

**Hypothesis tested:** Findings 01 and 02 left two confounded
explanations for the failure mode (memorization without
generalization): (a) SmolLM2-135M is too small, or (b) the recipe is
wrong. Finding 03 isolates these by training the **same audit-13
follow-up doc** on `qwen2.5-coder-1.5b` — same recipe, 11× the
parameters, code-pretrained.

**Verdict:** the recipe is the bottleneck. The bigger base eliminated
catastrophic forgetting (`cal_general` 26% → **0%** items regressed)
but the LoRA still memorizes instead of generalizes, and **actively
degrades the base on the wedge question we built the experiment to
answer**.

## Setup

- **Doc:** `docs/audits/13-followup/finding03/the-doc.dlm`
- **Store ULID:** `01KQDM14CT0X6AWFWTW9E81ZJ7`
- **Adapter:** `~/.dlm/store/01KQDM14CT0X6AWFWTW9E81ZJ7/adapter/versions/v0001`
- **Base:** `qwen2.5-coder-1.5b` (Qwen/Qwen2.5-Coder-1.5B-Instruct)
- **LoRA r/alpha:** 16/32 (same as Finding 01)
- **Steps:** 600 (same as Finding 01)
- **Final:** train 0.765, eval 0.750, mean token accuracy **82.2%** (vs 73.6% on the SmolLM2 stage-1 run)

## The pre-training baseline (`finding03/baseline.md`)

Before training, qwen2.5-coder-1.5b already speaks fortran fluently —
syntax, modern features (`do concurrent`, `intent`, `allocatable`),
program structure. What it does **not** know is `fortran-lang/stdlib`.
Q3 ("Show the signature of stdlib's sorting routine") makes the gap
visible: the base hallucinates `iso_fortran_env::sort`, a non-existent
module. **That hallucination is the wedge: a falsifiable claim our
LoRA could fix if domain knowledge transfer is real.**

## sway results

| probe | verdict | z | reading vs Finding 02 (SmolLM2-135M) |
| --- | --- | --- | --- |
| `dk_fortran` | FAIL | **−15.46σ** | LoRA shifts logits *less* than null on these prompts (similar to F02's −13.74σ) |
| `sis_fortran` | FAIL | ~0σ | 7/41 sections cleared, mean effective_sis = +0.000. Same uniform-bias collapse as F01. |
| `para_fortran` | ERROR | — | "no cases provided" — doc still lacks `!probe` markers |
| `leak_fortran` | PASS | **+7.71σ** | Memorization, fragility=0.00 (≈ F02's +6.41σ) |
| `cal_general` | **PASS** | — | **0/50 items regressed** (F02 had 26%, F01 had 10%) |
| `abl_fortran` | FAIL | — | overshoot=1.21 (vs F02's 1.49); R²=0.99 — linear loss surface, still under-saturated |

## Direct-query smoke (the wedge)

| Q | Base alone | Trained adapter | Net |
| --- | --- | --- | --- |
| Q1 (allocatable real64) | correct | correct | **= same** |
| Q2 (do concurrent) | partial | rambles + unrelated code | **worse** |
| Q3 (stdlib sorting — THE WEDGE) | wrong (hallucinates `iso_fortran_env::sort`) | degenerate enumeration loop | **worse** |
| Q4 (intent(in)) | correct | correct | **= same** |

**0/4 improved. 2/4 degraded.** The LoRA did not add stdlib knowledge;
it added autoregressive fortran-source-completion habits that
interfere with Q/A. Full transcripts at
`docs/audits/13-followup/finding03/direct-query-results.md`.

## What we learned

### 1. Base size is *not* the floor — it's the regularizer

Catastrophic forgetting collapsed from 26% (F02) → 10% (F01) → **0%
(F03)** as we moved from 135M to a bigger code-pretrained base. The
bigger base absorbs the LoRA without losing English chat capability.
This is a clean architectural finding:

> A LoRA adapter on a small base actively degrades the base. A LoRA
> adapter on a sufficiently capable base is "free" — the base carries
> through unchanged on what it already knew, and the LoRA contributes
> on top.

For dlm's product narrative: **the recommended base table needs a
warning at SmolLM2-135M.** That base is for style demos, not domain
knowledge addition.

### 2. The recipe is the bottleneck

With the bigger base eliminating the noise floor of catastrophic
forgetting, the *recipe-level* failures become unambiguous:

- `leak_fortran` +7.71σ — the LoRA memorizes raw fortran source
  fragments
- `dk_fortran` z=−15.46σ — the LoRA does not activate on Q/A-shaped
  prompts
- `sis_fortran` effective_sis=0 — uniform fortran-flavored bias, not
  per-section content
- Q3 wedge — the LoRA *makes the stdlib gap worse* by replacing a
  concrete-but-wrong base answer with a degenerate enumeration loop

These all share one root cause: **the training corpus is shaped like
raw source files, not Q/A pairs.** SFT's loss is "predict next token";
on raw source rows that means "complete fortran source," and the LoRA
learns exactly that. The 5MB FortranGoingOnForty source drowns the
1MB stdlib source (most of which exceeds the 32KB-per-file cap and is
silently skipped) and the 585KB of stdlib doc/specs markdown. The
LoRA never sees enough Q/A-shaped data to bind the format.

### 3. The adapter ablation curve has improved but not healed

Saturation overshoot dropped 1.49 (F02) → 1.21 (F03). R² rose to 0.99
(very linear). The trend says: with more training the curve might
saturate properly. But the loss is at 0.75 (eval), down from 0.95 →
0.6 across the 600 steps, and still descending. We're not at a real
minimum. **More steps would extract more memorization, not more
generalization.**

## Implications for the dlm product narrative

We now have a clean, falsifiable, evidence-backed claim:

> **dlm's value-add depends on training-data shape, not base capacity.**
> A LoRA trained on raw source code teaches the adapter to be a source
> autocomplete engine, regardless of base size. To produce a question-
> answering domain expert, the training data must itself be Q/A-shaped.

This is a *positive* product story, not a negative one — it tells dlm
users "use INSTRUCTION sections (or generate them from PROSE via
`dlm synth instructions`), not raw `sources:` directives, when the goal
is a chat assistant." The current docs don't make this distinction
clearly enough; that's a docs fix worth landing.

## Next experiment

[Finding 04 — Q/A-shaped training](./04-instruction-shape.md)
*(pending)*

The fix to test next: build a doc with INSTRUCTION sections constructed
from the stdlib `doc/specs/*.md` corpus (which is Q&A-friendly
documentation, unlike raw source files). Use `dlm synth instructions
--strategy extraction` with a strong teacher (claude-haiku, qwen-coder-7b,
or hand-author) to produce dense Q/A pairs. Train on qwen-coder-1.5b at
r=16 with no raw `sources:` directives. The wedge to track is still Q3:
does the trained adapter point at `stdlib_sorting` instead of producing
a degenerate loop?

If Q3 lands correctly, the dlm story is: bigger base + Q/A-shape recipe →
working domain expert. That's the experiment that *closes* the audit.
