---
dlm_id: 01KPKXHZNKB2J7KCRJ3B29J4H1
dlm_version: 1
base_model: smollm2-360m
system_prompt: |
  Generate changelog entries in the project's house style. Each entry
  is one sentence, imperative mood, past tense is banned.
training:
  adapter: lora
  lora_r: 4
  sequence_len: 1024
  num_epochs: 3
  learning_rate: 3e-4
  seed: 42
export:
  default_quant: Q4_K_M
  default_temperature: 0.2
---

# Changelog entry generator

Tiny utility `.dlm` — shows the daily-edit pattern. Every day you
paste a git log's `--oneline` output into `::instruction::` with the
clean entry below, retrain, and let the model phrase tomorrow's
commits in your voice.

## Style rules

- Imperative mood: "add", "fix", "refactor" — not "added", "fixes".
- One sentence per entry. No trailing period.
- Tag the area in parens: `(parser)`, `(cli)`, `(export)`.
- No "bump X to Y" — use "update X to Y".

::instruction::
### Q
feat(parser): support inline comments in frontmatter

### A
add inline-comment support to the frontmatter parser (parser)

### Q
fix(train): OOM guard regressed on grad_accum=1

### A
fix OOM guard when grad_accum is 1 so the recommendation bumps to 2 (train)

### Q
test(export): cover Q8_0 path in integration suite

### A
cover the Q8_0 export path in integration tests (export)

### Q
docs: add migration guide for legacy stores

### A
document the migration path for legacy stores (docs)
