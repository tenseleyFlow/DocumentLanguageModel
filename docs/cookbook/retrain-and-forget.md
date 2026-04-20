# Retrain without forgetting (and when to forget)

DLM's retention story: edits to a `.dlm` add new training signal while
previous versions of the document stay in a replay corpus. This recipe
walks through the default "additive" retrain, when to deliberately
forget, and how to audit what the model has seen.

## The default: additive retrain

```sh
# v0001 — initial training
$ uv run dlm train tutor.dlm
trained: v0001 (20 steps, seed=42, determinism=best_effort)
adapter: ~/.dlm/store/01KPM…/adapter/versions/v0001
log:     ~/.dlm/store/01KPM…/logs/train-000001-…jsonl

# Edit tutor.dlm — add more Q&A, fix a typo
$ $EDITOR tutor.dlm

# v0002 — retrain
$ uv run dlm train tutor.dlm
trained: v0002 (20 steps, seed=42, determinism=best_effort)
adapter: ~/.dlm/store/01KPM…/adapter/versions/v0002
log:     ~/.dlm/store/01KPM…/logs/train-000002-…jsonl
```

What `dlm train` does on retrain:

1. Loads the previous `manifest.json`; diffs this run's section hashes
   against the recorded `content_hashes`.
2. Appends the new/changed sections to the replay corpus under
   `replay/corpus.zst` (Sprint 08).
3. Builds the training set as `current sections + recency-weighted
   sample from replay`. Prior sections don't disappear — they keep
   showing up proportional to how recent they are.
4. Commits the new adapter as `v0002`; `adapter/current.txt` flips
   atomically (Sprint 09's two-phase commit).

The adapter from `v0002` still remembers v0001's Q&A pairs because the
training data for v0002 included them (via replay).

## Inspecting what the model has seen

```sh
$ uv run dlm show tutor.dlm
tutor.dlm
  dlm_id:         01KPM618S78XK668EX0TFEWAJY
  base_model:     qwen2.5-coder-1.5b (revision abcdef1)
  store:          ~/.dlm/store/01KPM…  (12.4 KB)
  adapter:        v0002
  training runs:  2 — last 2026-04-19T18:30:14
  exports:        0
```

Pretty-print summary; use `--json` for the full record including
`content_hashes`, `replay_size_bytes`, per-run `TrainingRunSummary`
entries, and `pinned_versions`:

```sh
$ uv run dlm show tutor.dlm --json | jq '.training_runs'
2
$ uv run dlm show tutor.dlm --json | jq 'keys'
[
  "adapter_version", "base_model", "base_model_revision",
  "content_hashes", "dlm_id", "exports", "has_adapter_current",
  "last_trained_at", "orphaned", "path", "pinned_versions",
  "replay_size_bytes", "source_path", "total_size_bytes",
  "training_runs"
]
```

## Intentional forgetting: `--fresh`

Sometimes you want the model to unlearn. Maybe the v0001 Q&A was
wrong, and replay-weighted retraining keeps reinforcing it.

```sh
$ uv run dlm train tutor.dlm --fresh
```

`--fresh` wipes the replay corpus AND the optimizer state; the next
run trains only on the current document's sections. The adapter
version still increments (`v0003`), and the manifest records that
this run was a fresh start.

Use sparingly — fresh training loses every prior training signal.

## Intentional pruning: edit the replay corpus

The replay corpus is a zstd-framed file on disk at
`<store>/replay/corpus.zst` + `<store>/replay/index.json`. There is
no shipped CLI for surgical pruning yet; drop into Python via
`dlm.replay.store.ReplayStore.at(corpus_path, index_path)` to read
or rewrite it. A first-class `dlm replay prune` command is on the
Phase 4 roadmap.

## What `dlm.lock` records

After each successful run, `dlm.lock` records:

- `last_run_id` — incremented each run
- `dlm_sha256` — hash of the source `.dlm` at this run
- `pinned_versions` — the tuple that produced this adapter

See [Determinism](../determinism.md) for the full field list and
mismatch policy.

## Caveats

- **Replay is lossy on eviction.** The corpus has a soft size cap
  (Sprint 08) and evicts the oldest, lowest-weight entries when it
  grows. Everything still in the corpus trains; evictions are lost.
- **`--fresh` doesn't forget the base model.** The base is whatever
  `base_model:` says in the frontmatter. Its pretraining data lives
  in its weights, not in your replay corpus.
