# Tokenized-section cache

When a `.dlm` ingests thousands of files via `training.sources`,
re-tokenizing everything on every `dlm train` run is the dominant
cost. The per-store tokenized-section cache avoids that: unchanged
files are retrieved from cache, only new or edited files hit the
tokenizer.

Target: second-run tokenization >5× faster than the first on a 1K-
file corpus. On a 50K-file corpus it's the difference between an
hour and tens of seconds.

## What gets cached

- **Directive-sourced sections only.** Files ingested via
  `training.sources` in the frontmatter. In-body sections
  (`::instruction::` fences) are cheap to tokenize and change more
  often, so they skip the cache.
- **Keyed by**: `(section_id, tokenizer_sha256, sequence_len)`. Any
  of the three changing invalidates the entry. Bump the base model
  (new tokenizer), bump the sequence length, or edit a file's
  content → that entry is gone.

## Layout

The cache lives under the per-store directory:

```
~/.dlm/store/<dlm_id>/tokenized-cache/
    manifest.json          version, tokenizer_sha256, total_bytes, entries
    entries/
        <section_id[:2]>/  sharded to avoid 50K files in one dir
            <key>.npz      numpy input_ids + attention_mask
```

`manifest.json` tracks per-entry metadata (size, last-access
timestamp) so LRU eviction doesn't need to stat every file.

## Inspecting the cache

```bash
dlm cache show /path/to/doc.dlm
# Cache for 01KPQ1FFEDGPPSMWRAS18SAZST
#   path:              ~/.dlm/store/01KP.../tokenized-cache
#   entries:           1,247
#   size:              312.4 MB
#   last-run hit rate: 98.4% (1228/1247)
```

Or machine-readable:

```bash
dlm cache show /path/to/doc.dlm --json | jq .
```

`dlm show --json` also reports `training_cache` at the top level.

## Maintenance

**Prune old entries** — drop anything not accessed in a cutoff:

```bash
dlm cache prune /path/to/doc.dlm --older-than 30d   # 30 days
dlm cache prune /path/to/doc.dlm --older-than 12h   # 12 hours
```

Default cutoff is `90d`. Stale entries accumulate after tokenizer
bumps or long breaks from a corpus — `prune` keeps disk bounded.

**Clear everything** — nuclear option:

```bash
dlm cache clear /path/to/doc.dlm
# confirms before deleting; pass --force to skip
```

## Tuning

The cache caps at **10 GiB by default**. A 50K-file corpus
tokenized at `sequence_len: 2048` is roughly 200 MB of int64 IDs, so
the default fits many codebases. LRU eviction keeps it bounded:
oldest-accessed entries go first, current-run entries are
protected (a cold cache won't self-starve).

Future knobs (not yet wired): `training.cache.max_bytes` in the
frontmatter will let a single `.dlm` tune its own cap. Until then,
caches are capped at the 10 GiB default.

## Invalidation triggers

| Trigger | Effect |
|---|---|
| File content edited | That section's `section_id` changes → new key, old entry orphaned (prune sweeps). |
| Tokenizer upgraded | `tokenizer_sha256` shifts → **every** entry for that family becomes unreachable. |
| `sequence_len` changed | All entries for that seq_len become unreachable. |
| Base model swapped | Usually bumps tokenizer → see above. |

Orphaned entries stay on disk until `prune` or `clear` removes them,
but `get()` never returns a stale entry — keys are exact.

## Measuring hit rate

The cache fires automatically during `dlm train` on any `.dlm` that
declares `training.sources`. To see how well it's working on your
corpus:

```bash
# After a training run, inspect per-run tokenization stats.
dlm show /path/to/doc.dlm --json | jq .training_cache
# {
#   "path": "~/.dlm/store/01KP.../tokenized-cache",
#   "entry_count": 1247,
#   "bytes": 327598080,
#   "last_run_hit_rate": 0.984,
#   "last_run_id": 3
# }
```

The metrics DB keeps a row per run:

```bash
dlm metrics /path/to/doc.dlm --json | jq '.runs[0].tokenization'
```

Fields on the event: `total_sections`, `cache_hits`, `cache_misses`,
`total_tokenize_seconds`, `cache_bytes_after`. Hit rate is
`cache_hits / (cache_hits + cache_misses)`.

**Reading the numbers.** A first run against a cold corpus is all
misses — that's the tokenize cost you pay once. Every subsequent run
against the same files should be all hits (rate → 1.0). If you see
hit rate drop unexpectedly on the second run, something invalidated
entries — check for a tokenizer upgrade (new `transformers`,
different base model revision) or a `sequence_len` change.

## Opting out

Some scenarios want the legacy tokenize-per-run path:

- debugging a suspected tokenization bug (is it the cache or the
  tokenizer?),
- cross-checking cached-vs-uncached determinism on the same seed.

The `--no-cache` flag on `dlm train` bypasses the cache for that run
without touching the on-disk entries:

```bash
dlm train /path/to/doc.dlm --no-cache
```

Entries from prior cached runs stay intact — the next run without
the flag picks them back up. No frontmatter change required.

## Pitfalls

- **Tokenizer upgrades invalidate the cache.** When you bump
  `transformers` or switch base models, expect one slow run while
  the cache re-warms. Pitfall #4 (the pad-token handling story)
  means you MUST NOT reuse tokens across tokenizer versions — the
  sha-based invalidation is the correctness barrier.
- **Not a shared cache.** Two `.dlm` files pointing at the same
  codebase tokenize twice. A future sprint may add cross-store
  deduplication; v1 keeps caches per-store for simplicity.
- **Disk pressure on huge corpora.** A 50K-file corpus at
  `sequence_len: 8192` can hit the 10 GiB cap quickly. Either trim
  via `--include` globs or plan for a `max_bytes` frontmatter knob
  in a follow-up sprint.

## What ships today

- Cache module (`dlm.directives.cache`) with atomic writes, LRU
  eviction, tokenizer-sha invalidation.
- `dlm cache show | prune | clear` CLI.
- Metrics wiring (`TokenizationEvent` → SQLite).
- `dlm show --json` surfaces cache state.
- Per-store layout under `<store>/tokenized-cache/`.
- **Trainer integration.** `dlm train` pre-tokenizes directive-
  sourced rows through the cache before handing a pre-processed
  dataset to TRL's `SFTTrainer`. Tokenizer output is bit-identical
  to TRL's own `_tokenize` path (guarded by an online parity test
  against the reference tokenizer).
- **`--no-cache` opt-out** on `dlm train` for debugging and
  determinism cross-checks.

Future follow-ups:

- **`training.cache.max_bytes` frontmatter knob** — per-`.dlm`
  tuning of the default 10 GiB cap (schema v8→v9).
- **Distributed / cross-store cache sharing** — explicit non-goal
  today.
