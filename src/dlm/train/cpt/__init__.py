"""Continued-pretraining refinements.

Three standalone utilities that the SFT trainer pulls in when the
document's training mix is dominated by prose (CPT) rows:

- `schedule.cosine_with_floor_lr` — LR curve tuned for DAPT (longer
  warmup, cosine decay to a floor instead of zero).
- `embed_warmup.unfreeze_embeddings_for` — context manager that
  unfreezes `embed_tokens` + `lm_head` for vocabulary warm-up.
- `vocab_gap.report` — tokenizer fit report (tokens-per-word, top
  unrepresented merges, `<unk>` hit count).
"""
