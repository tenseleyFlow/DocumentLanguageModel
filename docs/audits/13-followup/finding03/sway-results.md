# sway report

**Overall:** 0.47 (`partial`)  
**Base:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`  
**Adapter:** `adapter/versions/v0001`  
**Wall:** 86.7s  
**Determinism:** `best_effort` (seed=0)  
**Backend:** cache: 195/602 = 32% | batches: 6 (avg=5.3)  

## Components

| category | score | weight | |
|---|---:|---:|---|
| adherence | 0.01 | 0.30 |  |
| attribution | 0.50 | 0.35 |  |
| calibration | 0.88 | 0.20 |  |
| ablation | 0.79 | 0.15 |  |
| baseline | 1.00 | 0.00 | (informational, weight=0) |

## Probes

| name | kind | verdict | score | raw | ci95 | z | duration | note |
|---|---|---|---:|---:|---:|---:|---:|---|
| null_baseline | `null_adapter` | pass | 1.00 | — | — | — | 54.1s | null calibration: 5 kinds calibrated over 3 seeds (1 opted out) |
| dk_fortran | `delta_kl` | fail | 0.01 | 0.184 | [0.141, 0.228] | -15.46σ | 0.42s | mean js=0.1840, z=-15.46σ vs null |
| sis_fortran | `section_internalization` | fail | 0.50 | 0.000 | [-0.016, 0.016] | +0.00σ | 1.99s | 7/41 sections cleared; mean effective_sis=+0.000, z=+0.00σ vs null |
| para_fortran | `paraphrase_invariance` | error | — | — | — | — | 0.00s | no cases provided |
| leak_fortran | `leakage` | pass | 0.93 | 0.042 | [0.042, 0.042] | +7.71σ | 26.2s | greedy_recall=0.04 (perturbed=0.05, fragility=0.00), z=+7.71σ vs null |
| cal_general | `calibration_drift` | pass | 0.82 | 0.000 | [0.000, 0.000] | — | 3.00s | 0/50 items regressed >1.0 nats (frac=0.0%), mean_delta=-0.166 nats/tok (no calibration for calibration_drift) |
| abl_fortran | `adapter_ablation` | fail | 0.79 | 0.991 | — | — | 0.63s | R²=0.99, sat_λ=1.25 (out of band), overshoot=1.21 (no calibration for adapter_ablation) |

## Top findings

- dk_fortran (delta_kl) failed: mean js=0.1840, z=-15.46σ vs null
- adherence score is 0.01 — below the noise threshold
- 1 probe(s) errored — see full report for details

## Degenerate null calibration

1 probe kind(s) ran null_adapter but the resulting baseline was too narrow for z-scoring (std ≈ 0, typically `runs: 1` or coincidentally-matched seeds). Fix: bump `runs:` in the `null_adapter` spec entry. Affected kinds:

- `calibration_drift`
