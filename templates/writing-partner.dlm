---
dlm_id: 01KPKXHZNK08XAYPVA76AT5DBQ
dlm_version: 1
base_model: smollm2-1.7b
system_prompt: |
  Continue the passage in the author's voice. Prefer short sentences.
training:
  adapter: lora
  lora_r: 8
  sequence_len: 2048
  num_epochs: 2
  learning_rate: 1e-4
  seed: 42
export:
  default_quant: Q4_K_M
  default_temperature: 0.8
  default_top_p: 0.95
---

# Writing partner starter

Paste 20+ KB of your own prose below. The base model learns cadence,
vocabulary, and the idioms you reach for. Don't mix genres unless you
want the output to mix genres.

## Sample: short essays

Most architectural debates are linguistic debates in disguise. Two
people say "service" and mean different things. One means a process
with an HTTP endpoint; the other means a business capability owned
by a team. They argue for an hour before noticing.

Writing software is editing software. The first version is always a
draft. The trick is knowing when to stop polishing and ship.

The hardest part of a refactor isn't the refactor. It's convincing
yourself the old version was fine, so you don't feel obligated to
rewrite the rest of the codebase to match.

## Sample: weekly intros

**Week of 2025-12-08.** The holiday lull is a scam. Everyone slows
down, then January hits like a freight train with unrealized Q1
goals. I spent the week writing a migration plan I'd rather not have
written.

**Week of 2025-12-15.** Three conversations this week converged on
the same idea: "the simple version is almost always right, and the
simple version is almost always harder to find than the complex one."
