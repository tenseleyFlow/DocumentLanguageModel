# Save-to-train with `--watch`

`dlm train --watch` keeps the training context alive and re-runs an
incremental retrain every time you save the `.dlm` file. Pair it
with your editor and you get a feedback loop that turns authoring
into a conversation with the adapter.

## When to use it

- You're iterating on the content of a document and want each save
  to land immediately in the model.
- You're in an exploratory drafting phase — quick cycles matter
  more than full-dataset retrains.
- You want the training process to stay warm so cycles are seconds,
  not minutes.

Full retrains (no step cap, full dataset) still come from plain
`dlm train`. Watch mode is the drafting tool.

## Usage

```bash
dlm train mydoc.dlm --watch
```

That runs the normal initial train, then blocks on filesystem events.
Save the `.dlm` in your editor and the loop:

1. Coalesces rapid saves into a single trigger (`--watch-debounce-ms`,
   default 400 ms).
2. Reloads the doc and diffs it against `manifest.content_hashes`.
3. If no new sections: logs "no new content, skipping".
4. If new sections: runs `trainer.run(mode="resume",
   max_steps=<cap>)`. The cap (`--watch-max-steps`, default 100)
   keeps each cycle responsive.

## Flags

| Flag | Default | Effect |
|---|---|---|
| `--watch` | off | Enter save-to-train mode after the initial train. |
| `--watch-max-steps N` | 100 | Per-cycle step cap. Small so cycles take seconds. |
| `--watch-debounce-ms N` | 400 | Quiet interval before a burst of saves fires. |
| `--repl` | off | **Scaffolded only.** Threading bridge with `dlm repl` is a followup; the flag emits a clear refusal today. |

## Editor save patterns

Vim's `:w` writes a swap file then renames it atomically; VS Code
writes in place; Jupyter round-trips via an HTTP PUT. `watchfiles`
surfaces all three as modified events on the target path, and the
loop watches the parent directory + filename match so the rename
case doesn't drop events.

## Ctrl-C

- **Between cycles**: exits cleanly.
- **During a cycle**: the trainer owns the atomic commit; the
  current cycle completes (or the `training_state.pt` two-phase
  commit rolls it back), then the loop exits.

## Caveats

- **Laptop battery.** Watch mode doesn't sleep — each save spins
  the model up to its step cap. The default 100 steps on a tiny
  model is seconds; on a 3B model it's minutes. Reduce
  `--watch-max-steps` for bigger bases.
- **Concurrent editors.** Two editors writing the same `.dlm` can
  race the reload between cycles; the store lock catches it but
  you'll see "lock held" failures. Stick to one editor.
- **Full retrain for real releases.** Watch cycles are resume-mode
  + step-capped. When you're ready to promote an adapter, run
  `dlm train mydoc.dlm` (no `--watch`) so the full dataset flows
  through and a clean `dlm.lock` gets written.

## Deferred: REPL bridge

`--watch --repl` is on the DoD but marked `[~]`: the threading
between training and inference passes needs a test harness we
don't have in CI today. Follow the sprint file for when it lands.
Until then, run `dlm repl` in a second terminal while `--watch` is
running in the first — the store lock keeps them honest, and each
new adapter version becomes available to the REPL on its next load.
