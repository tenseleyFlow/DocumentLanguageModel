---
dlm_id: 01KQDGAM70EJ1WJCQY6PVDV95W
dlm_version: 15
base_model: smollm2-135m
training:
  sources:
    - path: ~/GithubOrgs/FortranGoingOnForty
      include: ["**/*.f90", "**/*.F90", "**/*.f95"]
      exclude:
        - "**/build/**"
        - "**/.git/**"
        - "**/dist/**"
      max_bytes_per_file: 32768
    - path: /tmp/stdlib_build/src
      include: ["**/*.f90", "**/*.F90", "**/*.fypp"]
      exclude:
        - "**/build/**"
        - "**/tests/**"
      max_bytes_per_file: 32768
    - path: /tmp/stdlib_build/doc/specs
      include: ["**/*.md"]
      max_bytes_per_file: 131072
  sources_policy: permissive
  adapter: lora
  lora_r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  sequence_len: 1024
  micro_batch_size: 1
  grad_accum: 8
  learning_rate: 1.5e-4
  warmup_ratio: 0.05
  num_epochs: 1
---

# Stage 1 — Fortran domain expansion (CPT)

Continual-pretraining stage of the two-stage Fortran expert recipe. This
.dlm has no INSTRUCTION sections — every training row is PROSE
(next-token loss on the raw fortran source + stdlib markdown specs).
Goal: expand the base model's fortran token-distribution prior. The
stage-2 .dlm will layer instruction-following on top of this adapter.

The corpus mirrors the prior follow-up doc but at higher LoRA rank
(r=64 vs r=16) so the adapter has the capacity to absorb the corpus
shift without hitting the saturation ceiling diagnosed by sway's
`adapter_ablation` overshoot.
