# Finding 02 — CPT-only stage isolates a *new* failure mode

**Hypothesis tested:** the original audit-13-followup adapter showed
"uniform fortran-flavored bias" without per-section content learning
(Finding 01). My theory: the PROSE corpus drowned the INSTRUCTION
signal. If we strip INSTRUCTION sections entirely and run pure CPT at
higher LoRA rank, the adapter should at least *learn the corpus* — and
we can layer Q/A on top in stage 2.

**Verdict:** the hypothesis was *partially* right (the adapter does
learn fortran tokens) but uncovered a stronger failure mode that
invalidates the two-stage chain idea on a 135M base.

## Setup

- **Doc:** `docs/audits/13-followup/stage1/the-doc.dlm`
- **Store ULID:** `01KQDGAM70EJ1WJCQY6PVDV95W`
- **Sources:** identical to Finding 01's run (FortranGoingOnForty + stdlib src + stdlib doc/specs markdown)
- **Sections:** PROSE-only, *zero* INSTRUCTION
- **LoRA r/alpha:** 64/128 (Finding 01 used 16/32)
- **Steps:** 800 (Finding 01 used 600; audit 13 used 120)
- **Final:** train 1.293, eval 1.223, mean token accuracy 73.6%

## sway results

| probe | verdict | z | reading |
| --- | --- | --- | --- |
| `dk_fortran_qa_shaped` | FAIL | **−13.74σ** | adapter shifts *less* than null on English-Q/A prompts |
| `sis_fortran` | SKIP | — | bridge needs ≥2 section kinds; PROSE-only doesn't qualify |
| `leak_fortran` | PASS | +6.41σ | memorized fragments (greedy_recall=0.06, fragility=0.00) |
| `cal_general` | FAIL | −4.81σ | **26%** general-comp items regressed >1 nat (was 10% in Finding 01) |
| `abl_fortran` | FAIL | — | overshoot=1.49, sat_λ=1.25 (out of band) |

## Direct-query smoke

Greedy (`temp=0.0`):

```
$ dlm prompt the-doc.dlm "module simple_demo\n  implicit none" --max-tokens 80
  implicit none
  implicit none
  implicit none
  implicit none
  ...        (24× repetition)

$ dlm prompt the-doc.dlm "How do you declare an allocatable real(real64) array?"
I am using the following code:
real(real64) array(1000000000, 1000000000)
I am using the following code:
real(real64) array(1000000000, 1000000000)
...        (loops on a single training fragment)
```

Sampling (`temp=0.7`, `top_p=0.9`):

```
$ dlm prompt the-doc.dlm "subroutine sort_real_array(arr, n)" --max-tokens 100
{
    arr = std::move(arr);
}
template <typename T>
void array_sort(T* arr, int n) {
    array_sort(arr, n);
}
```

— **C++ from a fortran prompt under sampling.** The base model's C++
prior dominates the moment we leave argmax decoding. The LoRA delta is
just memorized argmax tokens, not a generalized fortran prior.

## Diagnosis

Three signals together tell the architectural story:

1. **Mode collapse under greedy.** The adapter pushes a small set of
   fragments to the top of the distribution; argmax decoding traps in
   them. `leakage` z=+6.41σ + `fragility=0.00` is the textbook
   memorization fingerprint.
2. **Adapter inactive on non-fortran-shaped prompts.**
   `dk_fortran_qa_shaped` z=−13.74σ means the trained LoRA produces
   *less* divergence from base than a random LoRA on
   English-Q/A-shaped prompts. The training only taught the adapter
   to be active on raw-fortran-source-shaped inputs.
3. **General competence regressed twice as much as Finding 01.**
   `cal_general` 26% vs 10% items >1-nat regressed. Removing
   INSTRUCTION sections made things *worse* on English. The
   INSTRUCTION sections in the audit-13-followup were apparently
   acting as a chat-format regularizer.

The deepest read: **the LoRA learned form-specific completion (raw
fortran source autoregression), not domain knowledge.** This is
exactly what we asked SFT loss to do — every PROSE row's training
signal is "predict the next token of fortran source." We got what we
asked for. The token-distribution prior on `module x\n` is now sharper
toward fortran continuations, but the *concept* of fortran isn't
abstracted in a way that helps with English-prefixed questions.

## Why this invalidates the chain

The plan was: stage-1 CPT teaches fortran → stage-2 SFT layers Q/A.
The implicit assumption was that stage-1 produces a *fortran-aware
substrate* for stage-2 to bind onto. Instead stage-1 produced a
*memorization trap* that:

- Has degraded English chat capability (cal_general −4.81σ)
- Doesn't activate on English-prefixed prompts (dk z=−13.74σ)
- Mode-collapses under greedy decoding

A stage-2 SFT phase against this substrate would have to *un-do* the
memorization while teaching Q/A binding. SFT can't easily do both at
the same LoRA rank — and the ablation curve (overshoot=1.49, no
saturation in band) suggests the loss surface around the trained
point is linear, meaning the LoRA is still in "more is more" mode
rather than at a coherent minimum.

## What this means for the product narrative

dlm's promise is "edit a text file → trainable LLM." The architectural
reality from Findings 01 + 02:

- **At SmolLM2-135M:** the recipe consistently produces memorization,
  not generalization. Volume, rank, sequence length, and corpus
  shape variations have all bottomed out at the same failure mode.
- **The base model is the floor.** 135M params with our LoRA adapter
  can either preserve English-chat behavior *or* memorize fortran
  fragments — not compose both into a usable expert.

This is a *correct* and *useful* negative result for the product
narrative. It tells future users: don't expect a 135M base to absorb
specialty domains. It also informs dlm's recommended-base table —
the smollm2-135m row should carry a "use for style-transfer demos
only" caveat.

## Next experiment

[Finding 03](./03-base-floor.md) — promote the base to
qwen2.5-coder-1.5b (already registered in dlm). Test the question:
**is the recipe sound, and the 135M floor was the only blocker?** If
the same `the-doc.dlm` (audit-13-followup) on the bigger base
produces measurable LoRA delta beyond the base's existing fortran
knowledge, we have a working dlm story. If the LoRA delta is small,
we learn that dlm's value-add is style/format, not domain knowledge —
which is also publishable.
