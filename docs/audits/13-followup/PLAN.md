# Audit 13 follow-up — get fortran Q/A binding to work

**Goal:** flip `paraphrase_invariance` from **−3.51σ FAIL** to **PASS** and
get the Ollama smoke prompt to answer a fortran question with fortran (not
Python). Driven by Audit 13's diagnosis: the original .dlm had ~8 MB PROSE
source vs only **8** INSTRUCTION sections — too little Q/A signal for
SmolLM2-135M to bind question→answer semantics in 120 steps.

## Hypothesis

Q/A binding failed because of **corpus shape**, not corpus volume or model
floor. Three corrections:

1. **Add INSTRUCTION density.** Hand-author 50+ high-quality Q/A pairs
   sourced from the FORD-generated stdlib markdown specs at
   `/tmp/stdlib_build/doc/specs/` (40 expert-written module docs). Plus
   `dlm synth instructions --strategy extraction --apply` to harvest
   more Q/A from the PROSE.
2. **Add a third PROSE source.** Pull `/tmp/stdlib_build/doc/specs/**/*.md`
   in as source-directive content. The markdown files are "what does X do?"
   shaped — exactly the registration the model is missing.
3. **More steps.** 500-1000 instead of 120. Audit 13's loss curve was
   still descending at step 120 (1.927 → 1.638 over six log points).

If `paraphrase_invariance` still fails after this, the floor is real and we
need to repeat at SmolLM2-360M and 1.7B as a tier comparison.

## Sway spec for the re-run

Will require `pip install 'dlm-sway[hf,dlm]'` in the audit venv so the
bridge lights up `section_internalization`, `leakage`, and bridge-aware
`paraphrase_invariance` (Audit 13 m13.8).

```yaml
version: 1
dlm_source: ./the-doc.dlm
models:
  base: { kind: hf, base: "HuggingFaceTB/SmolLM2-135M-Instruct" }
  ft:   { kind: hf, base: "HuggingFaceTB/SmolLM2-135M-Instruct",
          adapter: "~/.dlm/store/<ULID>/adapter/versions/v0001" }
defaults:
  seed: 0
  differential: true
  coverage_threshold: 0.6
  score_weights: { adherence: 0.30, attribution: 0.35,
                   calibration: 0.20, ablation: 0.15 }
suite:
  - { name: null_baseline, kind: null_adapter, prompts_from: sections/instruction }
  - { name: dk_fortran, kind: delta_kl, prompts_from: sections/instruction }
  - { name: para_fortran, kind: paraphrase_invariance,
      prompts_from: sections/instruction,
      assert: { generalization_ratio_gte: 0.5 } }
  - { name: sis_fortran, kind: section_internalization }    # bridge probe
  - { name: leak_fortran, kind: leakage,                    # bridge probe
      assert: { fragility_gte: 0.4 } }
  - { name: cal_general, kind: calibration_drift,
      assert: { regression_rate_lt: 0.15 } }
  - { name: abl_fortran, kind: adapter_ablation,
      lambdas: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
      assert: { r_squared_gte: 0.85 } }
```

## Success bar

- `paraphrase_invariance` z-score ≥ +1.0 (PASS), generalization_ratio ≥ 0.5.
- `section_internalization` ≥ +1.0σ on at least 5 of the new INSTRUCTION
  sections (bridge probe — verifies attribution).
- `leakage` `fragility_gte ≥ 0.4` (sanity: pattern-match, not memorization).
- `delta_kl` z-score ≥ +1.0 on a held-out fortran prompt set (the audit-13
  +44σ was huge; we just need real signal, not regression).
- `adapter_ablation` R² ≥ 0.85 on the λ-scaled curve (healthy fine-tune).
- Ollama smoke: "When should I use `do concurrent`?" → fortran answer with
  `do concurrent` syntax visible.

If any of these miss, the next iteration is a model-floor sweep at
SmolLM2-360M and 1.7B with the same recipe. If they all pass at 135M, the
recipe is the dlm cookbook starter.
