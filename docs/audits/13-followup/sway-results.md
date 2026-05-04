# sway report

**Overall:** 0.57 (`partial`)  
**Base:** `HuggingFaceTB/SmolLM2-135M-Instruct`  
**Adapter:** `adapter/versions/v0001`  
**Wall:** 83.5s  
**Determinism:** `best_effort` (seed=0)  
**Backend:** cache: 195/602 = 32% | batches: 6 (avg=5.3)  

## Components

| category | score | weight | |
|---|---:|---:|---|
| adherence | 0.49 | 0.30 |  |
| attribution | 0.50 | 0.35 |  |
| calibration | 0.67 | 0.20 |  |
| ablation | 0.79 | 0.15 |  |
| baseline | 1.00 | 0.00 | (informational, weight=0) |

## Probes

| name | kind | verdict | score | raw | ci95 | z | duration | note |
|---|---|---|---:|---:|---:|---:|---:|---|
| null_baseline | `null_adapter` | pass | 1.00 | — | — | — | 52.0s | null calibration: 5 kinds calibrated over 3 seeds (1 opted out) |
| dk_fortran | `delta_kl` | fail | 0.49 | 0.151 | [0.132, 0.167] | -0.15σ | 0.33s | mean js=0.1511, z=-0.15σ vs null |
| sis_fortran | `section_internalization` | fail | 0.50 | -0.000 | [-0.020, 0.021] | -0.00σ | 1.70s | 7/41 sections cleared; mean effective_sis=-0.000, z=-0.00σ vs null |
| para_fortran | `paraphrase_invariance` | error | — | — | — | — | 0.00s | no cases provided |
| leak_fortran | `leakage` | pass | 0.94 | 0.061 | [0.061, 0.061] | +8.45σ | 25.7s | greedy_recall=0.06 (perturbed=0.05, fragility=0.17), z=+8.45σ vs null |
| cal_general | `calibration_drift` | pass | 0.40 | 0.100 | [0.040, 0.200] | — | 2.97s | 5/50 items regressed >1.0 nats (frac=10.0%), mean_delta=-0.259 nats/tok (no calibration for calibration_drift) |
| abl_fortran | `adapter_ablation` | fail | 0.79 | 0.915 | — | — | 0.63s | R²=0.91, sat_λ=1.25 (out of band), overshoot=1.54 (no calibration for adapter_ablation) |

## Top findings

- dk_fortran (delta_kl) failed: mean js=0.1511, z=-0.15σ vs null
- 1 probe(s) errored — see full report for details

## Degenerate null calibration

1 probe kind(s) ran null_adapter but the resulting baseline was too narrow for z-scoring (std ≈ 0, typically `runs: 1` or coincidentally-matched seeds). Fix: bump `runs:` in the `null_adapter` spec entry. Affected kinds:

- `calibration_drift`
