# Audit 13 follow-up — findings

**Verdict:** the corpus-shape hypothesis was *not* falsified, but the
re-run produced an **adapter that memorizes more and generalizes less**
than audit 13's smaller adapter. Five-fold INSTRUCTION density + 5×
steps moved the failure from "no Q/A binding" (audit 13) to "verbatim
memorization without prompt-level behavior change" (this run).

## Setup

- **Doc:** `docs/audits/13-followup/the-doc.dlm`
- **Store ULID:** `01KQCWMA64901VEMYB3DC8CGXY`
- **Adapter:** `~/.dlm/store/01KQCWMA64901VEMYB3DC8CGXY/adapter/versions/v0001`
- **Steps:** 600 (audit 13 = 120)
- **LoRA r/alpha:** 16/32 (audit 13 = 8/16)
- **Sources:** `~/GithubOrgs/FortranGoingOnForty` + `/tmp/stdlib_build/src` +
  `/tmp/stdlib_build/doc/specs` markdown
- **Final loss:** train 1.452, val 1.401 (audit 13 = 1.83 at step 120, still descending)

## sway results, side-by-side

| probe | audit 13 (baseline) | follow-up (this run) | delta |
| --- | --- | --- | --- |
| `delta_kl` | **PASS, z=+44.16σ** | **FAIL, z=−0.15σ** | massive regression |
| `section_internalization` | SKIP (no `[dlm]`) | **~0, z≈0** | weak signal |
| `paraphrase_invariance` | FAIL, z=−3.51σ | ERROR ("no cases provided") | unresolved |
| `leakage` | SKIP (no `[dlm]`) | **PASS, z=+8.45σ** | strong memorization |
| `calibration_drift` | PASS | borderline (5/50 regressed) | mild degradation |
| `adapter_ablation` | PASS (R²=0.91) | **FAIL, overshoot=1.54** | failed sat-band |

## Diagnosis

### Why `delta_kl` collapsed
Audit 13's small adapter on the same prompts moved logits much more
than its null-baseline counterpart (z=+44σ). This run's mean JS
divergence on the prompts is `0.151` — a meaningful number in
absolute terms, but the null-adapter calibration baseline is *also*
≈0.15 because random LoRA weights at r=16 already perturb logits a
lot on short fortran prefix prompts. The trained adapter doesn't
*beat noise* on these specific prompt prefixes.

This is calibration-shape sensitivity, not "the adapter learned
nothing." Different prompt selection would surface the signal.

### Why `section_internalization` is ≈0 (this is the load-bearing finding)
Per-section evidence in `/tmp/sway-13fu.json`:

- `own_lift` (NLL improvement on the section's own probes) ≈ 0.030
- `leak_lift` (NLL improvement on *other* sections' probes) ≈ 0.064
- `effective_sis = own_lift − leak_lift` ≈ **−0.034**

`leak_lift` is *uniform* across sections at ~0.064. That means the
adapter applied a **constant fortran-flavored prior** to every
fortran-shaped prompt — not section-specific knowledge. The adapter
learned "this looks like fortran, lower NLL" but did **not** learn
"section X said Y, so on probes about Y, lower NLL more."

That's the corpus-shape issue surfacing differently. With 40 INSTRUCTION
sections that mostly differ only in surface wording, the adapter
reduced to a generic fortran-style bias.

### Why `leakage` jumped to +8.45σ
greedy LCS recall against perturbed prompts is high (0.06 on
perturbed vs 0.05 baseline; fragility 0.17). Combined with weak
`section_internalization`, this is the textbook **memorization**
signature: the model can recite chunks of training data when prompted,
but the knowledge isn't transferable.

### Why `paraphrase_invariance` errored ("no cases provided")
The bridge's case generator wants paired (original, paraphrased) probe
items, and it got none. Likely cause: the audit follow-up `.dlm` puts
INSTRUCTION sections in compact `Q:`/`A:` shape but without the `!probe`
markers that the bridge's section→probes mapper looks for. (The
parser-expansion PR #10 fixed *parsing* of these blocks; it did not add
`!probe` markers to the audit doc.)

This isn't a sway bug — it's a doc-shape gap. The follow-up doc was
authored before `dlm synth instructions --apply` was used.

### Why `adapter_ablation` overshot (R²=0.91, sat_λ=1.25, overshoot=1.54)
λ-scaled KL: at λ=1.25 KL is 1.54× the λ=1.0 KL. The healthy band is
overshoot ≤1.05 with sat_λ ∈ (0.5, 1.0]. This curve never saturated —
which mathematically reads as **the adapter is under-magnitude**:
scaling its contribution beyond the nominal training point keeps
pulling logits toward the same direction. R²=0.91 is fine (linear
response is healthy); the problem is the *band*, not the shape.

This often co-occurs with under-training rather than over-training.
With a memorization signature elsewhere though, the more likely read
is *low-rank knowledge that the rest of the model can't compose with* —
the LoRA delta is small, the rest of the model produces fortran-flavored
output by base capability, and λ-scaling the small delta linearly
amplifies its directional bias.

## Direct query smoke test

```
$ uv run --no-sync dlm prompt the-doc.dlm \
    "How do you declare an allocatable array of real(real64) in modern Fortran?" \
    --max-tokens 120 --temp 0.0
I am using the following code:
real(real64) array(1000000000, 1000000000)
I am using the following code:
real(real64) array(1000000000, 1000000000)
...
```

The adapter clearly learned fortran-shaped output (it uses `real(real64)`,
correct kind syntax) — but did not learn the *answer pattern* for a
typed Q/A query. It loops on a single memorized fragment.

## Implications for the audit

1. **Original audit 13 finding (`paraphrase_invariance` FAIL on 8
   instruction sections) was load-bearing.** Adding 5× more sections
   without changing their *shape* worsened generalization. The fix
   wasn't volume — it was Q/A shape diversity (different question
   forms for the same content).

2. **Null-adapter calibration is prompt-sensitive.** A trained adapter
   at r=16 needs prompts where the null adapter produces *less* JS
   divergence — i.e. prompts with strong base-model priors. Short
   fortran-fence prefixes don't qualify.

3. **The bridge `paraphrase_invariance` probe needs `!probe` markers
   on instruction sections** (or sway should derive them). Without
   markers + paraphrases, the bridge silently becomes "no cases."

## Recommended next steps (not started)

- **Generate paraphrases.** `dlm synth instructions --strategy paraphrase
  --per-section 3 --apply` against the existing 40 INSTRUCTION sections,
  then re-run sway. This populates the bridge's paraphrase cases and
  also breaks the surface-form uniformity that drove `effective_sis ≈ 0`.
- **Tier comparison at SmolLM2-360M and 1.7B.** If 600 steps + 40 sections
  + paraphrases still doesn't bind Q/A on 135M, the floor is the model.
- **Adjust calibration prompts.** For `delta_kl`, swap the fortran-fence
  prefixes for prompts where SmolLM2 has a strong *English* prior
  (e.g. "When should you prefer `do concurrent` over `do`? Answer:") so
  the null baseline produces low JS divergence and the trained adapter's
  divergence shows up against noise.
- **Don't claim the e2e fortran promise yet.** This run shows the .dlm →
  trainable adapter pipeline works (training completed deterministically,
  adapter saved, sway runs end-to-end with bridge probes lit) — but it
  also shows that *the right adapter* requires more corpus engineering
  than "more sections + more steps." That's a real product caveat.
