# sway report

**Overall:** 0.55 (`partial`)  
**Base:** `HuggingFaceTB/SmolLM2-135M-Instruct`  
**Adapter:** `adapter/versions/v0001`  
**Wall:** 73.6s  
**Determinism:** `best_effort` (seed=0)  
**Backend:** cache: 180/558 = 32% | batches: 6 (avg=4.7)  

## Components

| category | score | weight | |
|---|---:|---:|---|
| adherence | 0.52 | 0.30 |  |
| attribution | 0.50 | 0.35 |  |
| calibration | 0.50 | 0.20 |  |
| ablation | 0.79 | 0.15 |  |
| baseline | 1.00 | 0.00 | (informational, weight=0) |

## Probes

| name | kind | verdict | score | raw | ci95 | z | duration | note |
|---|---|---|---:|---:|---:|---:|---:|---|
| null_baseline | `null_adapter` | pass | 1.00 | — | — | — | 48.7s | null calibration: 5 kinds calibrated over 3 seeds (1 opted out) |
| dk_fortran_qa_shaped | `delta_kl` | fail | 0.52 | 0.219 | [0.097, 0.359] | +0.23σ | 0.16s | mean js=0.2191, z=+0.23σ vs null |
| sis_fortran | `section_internalization` | fail | 0.50 | 0.000 | [-0.043, 0.045] | +0.00σ | 1.39s | 15/36 sections cleared; mean effective_sis=+0.000, z=+0.00σ vs null |
| para_fortran | `paraphrase_invariance` | error | — | — | — | — | 0.00s | no cases provided |
| leak_fortran | `leakage` | pass | 0.66 | 0.031 | [0.031, 0.031] | +2.04σ | 19.0s | greedy_recall=0.03 (perturbed=0.04, fragility=0.00), z=+2.04σ vs null |
| cal_general | `calibration_drift` | pass | 0.34 | 0.120 | [0.040, 0.220] | — | 3.11s | 6/50 items regressed >1.0 nats (frac=12.0%), mean_delta=-0.170 nats/tok (no calibration for calibration_drift) |
| abl_fortran | `adapter_ablation` | fail | 0.79 | 0.910 | — | — | 1.06s | R²=0.91, sat_λ=1.25 (out of band), overshoot=1.88 (no calibration for adapter_ablation) |

## Top findings

- dk_fortran_qa_shaped (delta_kl) failed: mean js=0.2191, z=+0.23σ vs null
- 1 probe(s) errored — see full report for details

## Degenerate null calibration

1 probe kind(s) ran null_adapter but the resulting baseline was too narrow for z-scoring (std ≈ 0, typically `runs: 1` or coincidentally-matched seeds). Fix: bump `runs:` in the `null_adapter` spec entry. Affected kinds:

- `calibration_drift`
