# Metrics & observability

Every `dlm train` cycle writes its step and eval metrics into a
per-store SQLite database at `~/.dlm/store/<dlm_id>/metrics.sqlite`.
`dlm metrics` reads from that DB. Optional TensorBoard / W&B sinks
are available behind the `observability` extra.

## What gets recorded

- **runs**: one row per `trainer.run` invocation —
  `run_id`, `started_at`, `ended_at`, `adapter_version`, `phase`,
  `seed`, `status` (`running` / `ok` / `failed` / `cancelled`).
- **steps**: one row per logged training step —
  `run_id`, `step`, `loss`, `lr`, `grad_norm`, timestamp.
- **evals**: one row per eval cadence hit —
  `run_id`, `step`, `val_loss`, `perplexity`, optional `retention`.
- **exports**: one row per `dlm export` completion.

Writes are best-effort: a metrics failure never takes down training.

## `dlm metrics <path>`

Default view lists the most-recent runs:

```bash
$ dlm metrics mydoc.dlm
Runs: 3
  run_id=3  phase=sft  seed=42  status=ok  started=2026-04-20T17:12:04Z
  run_id=2  phase=sft  seed=42  status=ok  started=2026-04-20T16:58:11Z
  run_id=1  phase=sft  seed=42  status=ok  started=2026-04-20T16:40:22Z
```

Drill into one run with `--run-id N` to see step + eval counts.
`--json` emits a machine-readable object; `--csv` emits the steps +
eval table for spreadsheet import.

### Filters

- `--phase sft|dpo|orpo|cpt` — restrict to one training phase.
- `--since 24h|7d|30m|10s` — time window on `started_at`.
- `--run-id N` — drill-down on a specific run.
- `--limit N` — cap the number of runs returned (default 20).

## `dlm metrics watch <path>`

Tails the metrics DB — prints new step and eval rows as they land.
Useful in a second terminal while `dlm train` (or `dlm train --watch`)
runs in the first.

```bash
$ dlm metrics watch mydoc.dlm
metrics watch: polling ~/.dlm/store/01HZ.../ every 1.0s (Ctrl-C to exit)
→ following run_id=4
  step    10  loss=1.87  lr=0.0002  grad_norm=0.31
  step    20  loss=1.73  lr=0.00018  grad_norm=0.27
  eval @ step 20  val_loss=1.81  perplexity=6.11
```

`--poll-seconds N` tunes how often the DB is re-read (default 1.0).

## TensorBoard sink

```bash
uv sync --extra observability
dlm train mydoc.dlm --tensorboard
tensorboard --logdir ~/.dlm/store/<dlm_id>/tensorboard
```

The sink writes one run directory per `trainer.run` under
`store/tensorboard/run_NNNN/`. Scalars logged: `train/loss`,
`train/lr`, `train/grad_norm`, `eval/val_loss`, `eval/perplexity`.

Skipped cleanly if the `observability` extra isn't installed — you
get the SQLite DB either way.

## W&B sink (opt-in)

```bash
uv sync --extra observability
dlm train mydoc.dlm --wandb my-project
```

Runs W&B in **offline mode** by default. The run directory sits at
`store/wandb/offline-run-*/`. To upload, run `wandb sync <dir>`
explicitly — we never upload automatically. If you haven't logged
in to W&B, offline mode still captures the run locally for later
review.

Privacy posture: no network calls from the training process.
Uploading is always a separate, explicit step.

## SQLite schema

The database at `metrics.sqlite` is queryable directly:

```bash
sqlite3 ~/.dlm/store/<dlm_id>/metrics.sqlite
sqlite> .tables
evals    exports  runs     steps
sqlite> SELECT run_id, phase, status FROM runs;
```

WAL mode is on: readers (including the CLI) don't block the trainer,
and a Ctrl-C mid-write leaves a recoverable DB.

## Pruning

No auto-prune today. If the DB grows past comfort, drop older rows:

```sql
DELETE FROM steps WHERE run_id NOT IN (SELECT run_id FROM runs ORDER BY run_id DESC LIMIT 10);
DELETE FROM evals WHERE run_id NOT IN (SELECT run_id FROM runs ORDER BY run_id DESC LIMIT 10);
DELETE FROM runs  WHERE run_id NOT IN (SELECT run_id FROM runs ORDER BY run_id DESC LIMIT 10);
VACUUM;
```

A built-in `dlm metrics prune` is on the backlog.
