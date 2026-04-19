# Writing partner

Train a model on prose you've already written so it can continue in
your voice. Useful for drafting newsletter intros, email replies,
tech-blog paragraphs.

## Goal

Heavy-prose document, minimal Q&A. The model learns cadence, verb
choice, sentence length, and the specific idioms you reach for.

## Template

```dlm
---
dlm_id: 01HRWRITER0000000000000000
base_model: smollm2-1.7b
system_prompt: |
  Continue the passage in the author's voice. Prefer short sentences.
training:
  lora_r: 8
  sequence_len: 2048
  num_epochs: 2
  learning_rate: 1e-4      # softer — style, not content
export:
  default_quant: Q4_K_M
  default_temperature: 0.8   # creative continuation
  default_top_p: 0.95
---

# Sample essays

Most architectural debates are linguistic debates in disguise. Two
people say "service" and mean different things. One means a process
with an HTTP endpoint; the other means a business capability owned
by a team. They argue for an hour before noticing.

Writing software is editing software. The first version is always a
draft. The trick is deciding when to stop polishing and ship.

...

# Newsletter intros

**Week of 2025-12-08.** The holiday lull is a scam. Everyone slows
down, then January hits like a freight train with unrealized Q1
goals. I spent the week writing a migration plan I'd rather not have
written.

**Week of 2025-12-15.** Three conversations this week converged on
the same idea: "the simple version is almost always right, and the
simple version is almost always harder to find than the complex one."
```

## Walk-through

```sh
$ uv run dlm init writer.dlm --base smollm2-1.7b
$ # Paste essays + newsletter drafts under a few prose headings
$ uv run dlm train writer.dlm
$ uv run dlm prompt writer.dlm --temp 0.8 "Week of 2026-01-05. I've been thinking about"
Week of 2026-01-05. I've been thinking about how much of product work
is just sitting with a question long enough that the answer becomes
obvious in hindsight…
```

## Prose length rule of thumb

- **Under 20 KB of prose**: style doesn't really take. Add more.
- **50–200 KB**: converges nicely in 2 epochs; generations feel
  recognizably yours.
- **Over 500 KB**: training wall-clock grows linearly; the replay
  corpus handles this fine, but start with a smaller cut and add
  on retrains.

## Tips

- Keep prose in logical sections — one essay per `#` heading. The
  parser treats them as one big prose section but humans reading the
  file thank you later.
- Don't mix genres unless you want the model to mix genres. A `.dlm`
  with both legal memos and jokes produces generations that can't
  decide which to be.
- The `temperature: 0.8` in the export frontmatter is a hint for
  `ollama run` — `dlm prompt` reads `--temp` from the CLI and ignores
  this.
