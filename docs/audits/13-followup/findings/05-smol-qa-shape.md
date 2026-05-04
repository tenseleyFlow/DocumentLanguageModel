# Finding 05 — Q/A-shape recipe on smol: floor still bites

**Hypothesis tested:** Finding 04 closed the wedge with a Q/A-shape
recipe on `qwen2.5-coder-1.5b`. Finding 02 ruled SmolLM2-135M
inadequate using a *raw-source* (PROSE-only) recipe. Open question:
was finding 02's verdict really about base size, or about the
combination of small base + bad recipe? Take finding 04's exact
INSTRUCTION-only corpus, swap in `smollm2-135m`, run.

**Verdict:** **architectural floor confirmed.** The recipe-shape fix
is necessary but not sufficient. SmolLM2-135M overfits the 32 train
pairs, fails to generalize, and damages base capability — all signals
qualitatively worse than finding 02's PROSE recipe on the same base.
Finding 02's "use 135M for style-transfer demos only" caveat
generalizes: it's a base-size constraint, not a recipe constraint.

## Setup

- **Doc:** `docs/audits/13-followup/finding05-smol-qa/the-doc.dlm`
- **Store ULID:** `01KQGBY1QWPFBKE0E8487PRB3E`
- **Adapter:** `~/.dlm/store/01KQGBY1QWPFBKE0E8487PRB3E/adapter/versions/v0001`
- **Base:** `smollm2-135m`
- **Sections:** 35 INSTRUCTION (32 train / 3 val), copied verbatim
  from finding 04 — *only* base + dlm_id changed
- **Recipe:** r=16 / α=32 / lr=2e-4 / 100 epochs (early-stopped at epoch 60)

## Training curve

| Epoch | train_loss | eval_loss | train_acc | eval_acc |
|---:|---:|---:|---:|---:|
| 10 |  2.62 | 2.22 | 0.52 | 0.61 |
| 20 |  1.77 | 1.64 | 0.65 | **0.68** ← best |
| 30 |  1.29 | 1.57 | 0.74 | 0.66 |
| 40 |  0.86 | 1.96 | 0.84 | 0.64 |
| 50 |  0.55 | 2.18 | 0.91 | 0.65 |
| 60 |  0.31 | 2.37 | 0.95 | 0.63 |

**Eval bottoms at epoch 20-30 then climbs while train continues to
descend** — textbook overfitting. Finding 04 same recipe on
qwen-coder-1.5b: final eval **0.047**, eval_acc **0.986**. Smol's
endpoint eval is 50× higher; held-out token-acc is 35 percentage
points lower. The base can memorize the train pairs but lacks the
parameter capacity to generalize the patterns.

## sway results

Composite **0.55 (`partial`)**. Full report at
[`finding05-smol-qa/sway-results.md`](../finding05-smol-qa/sway-results.md).

| probe | verdict | z | reading |
|---|---|---:|---|
| `dk_fortran_qa_shaped` | **FAIL** | +0.23σ | adapter shift on Q/A prompts is noise-level; the Q/A-shape recipe didn't even teach smol to *fire* on Q/A-shaped inputs |
| `sis_fortran` | **FAIL** | +0.00σ | 15/36 sections cleared; **no per-section internalization signal whatsoever** |
| `para_fortran` | ERROR | — | sway bridge "no cases provided" — separate sway issue |
| `leak_fortran` | PASS | +2.04σ | greedy_recall=0.03, fragility=0.00 — textbook memorization fingerprint |
| `cal_general` | pass* | — | **12% items regressed >1 nat** (was 26% in finding 02); recipe-shape fix mitigates forgetting but doesn't eliminate it |
| `abl_fortran` | **FAIL** | — | R²=0.91 linearity, but **overshoot=1.88, sat_λ=1.25 out of band** — adapter is in pathological "more is more" territory, not converged at a coherent minimum |

*`cal_general` formally passed because its null-baseline std collapsed
to zero (1 of the 3 seeds duplicated), but the raw 12% regression is
the load-bearing signal — that's 6/50 general-competence items broken
by the LoRA on a model that started at 100%.

The adherence and attribution probes both at noise-level (z ≈ 0) is
the cleanest signal here: **the adapter doesn't reliably activate on
the trained input shape** despite 95% train token-acc. That's the
parameter-capacity bottleneck on display.

## Direct-query smoke

Full transcripts at
[`finding05-smol-qa/direct-query-results.md`](../finding05-smol-qa/direct-query-results.md).
Highlights vs finding 04:

| Query | Finding 04 (qwen-coder-1.5b) | Finding 05 (smol) |
|---|---|---|
| Wedge (sorting signature) | clean verbatim | partial verbatim → gibberish |
| Held-out same-module | partial generalization | token salad |
| Held-out different module | plausible wrong API | non-syntactic hallucination |
| Capital of France | unchanged from base | "Fortified AI"/"NAM module" bleed |
| **2 + 2** | (not tested) | "you're calling `stdlib_array_plus`" |

The 2+2 result is the punchline: the LoRA so saturates the small base
that even arithmetic gets routed through fortran-domain hallucinations.
Finding 02 measured this as `cal_general` 26% items regressed >1 nat;
the qualitative picture is consistent.

## Why this falsifies the "recipe-shape rescues smol" idea

Finding 04 narrative: "raw source teaches autocomplete; INSTRUCTION
teaches Q/A". One way to read this: the bad recipe was masking a base
that could in principle handle either domain expansion or chat — just
not both at once via raw-source training.

Finding 05 falsifies this: with the *cleanest* recipe (no raw sources,
INSTRUCTION-only, exactly the corpus that worked on a 1.5B base), smol
*still* destroys general capability. The recipe wasn't the bottleneck;
the parameter count is. Going from "memorization with raw sources" to
"memorization with Q/A pairs" doesn't buy generalization, it just
changes what gets memorized.

This is the cleanest signal in the audit for the **base-size table**
in dlm's docs: the smollm2-135m row should refuse domain training in
recommended configurations and steer users to ≥ 1B for any
specialty-knowledge task.

## What this confirms about finding 02

Finding 02's MLX-fix retest (appended 2026-04-30) already showed the
adapter was being applied during finding 02; the negative result wasn't
an MLX silent-bypass artifact. Finding 05 closes the loop: even with
the recipe correction (which would have been the next obvious thing to
try if MLX-bypass were the issue), the verdict holds. The
architectural-floor finding is robust across:

- Recipe shape (PROSE-only in finding 02 vs INSTRUCTION-only in
  finding 05)
- Inference backend (verified MLX & PyTorch in finding 02 retest)
- Training duration (60 epochs early-stopped here, 800 steps in
  finding 02)

## Implications for the dlm product narrative

Updating the three-step story from finding 04:

1. **Use a base ≥ 1B params.** Smaller bases (135M) actively degrade
   under LoRA training of *any* shape. Finding 05 confirms this is a
   floor, not a recipe-fixable failure. Document a hard refusal in
   `dlm doctor` or a loud warning at `dlm train` time when the
   selected base is below the floor and the corpus is non-trivial.
2. **Train on INSTRUCTION-shaped data, not raw source code.**
   (Unchanged from finding 04.)
3. **Plan one Q/A pair per question you want to answer.**
   (Unchanged from finding 04, but with the caveat that this only
   works above the base-size floor.)

## Next experiment (optional)

The base-size *floor* is now diagnosed; the question of where the
ceiling sits is open but lower priority for the product story.
Possibilities for a finding 06 or future audit:

- Test SmolLM2-360M and SmolLM2-1.7B with the same finding-05 corpus —
  is the cliff between 135M and 1B continuous or stepped?
- Test qwen2.5-coder-0.5B vs 1.5B with the same corpus — does
  code-pretraining lower the floor by domain proximity?

Both are nice-to-have for the recommended-base table refinement; the
finding-04+05 pair is sufficient to make the product claim ("use ≥ 1B")
publishable as written.
