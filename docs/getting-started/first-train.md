# First training cycle

This walks you through creating a `.dlm` document, training a LoRA
adapter against `smollm2-135m`, and confirming the artifacts on disk.

## 1. Create a document

```sh
$ uv run dlm init tutor.dlm --base smollm2-135m
created: tutor.dlm
dlm_id: 01KC…                (26-character ULID)
base:   smollm2-135m         (HuggingFaceTB/SmolLM2-135M-Instruct)
store:  ~/.dlm/store/01KC…/
```

`dlm init` writes a minimal `.dlm` with a fresh ULID in the frontmatter
and provisions the store directory.

Open `tutor.dlm` in your editor and add some training signal:

```dlm
---
dlm_id: 01KC...
dlm_version: 1
base_model: smollm2-135m
training:
  seed: 42
---

# Python decorators primer

::instruction::
### Q
What is a Python decorator?

### A
A decorator is a function that takes another function as input and
returns a new function that wraps extra behavior around the original.
The `@decorator_name` syntax above a `def` is equivalent to
`name = decorator_name(name)`.

### Q
When should I use `functools.wraps`?

### A
Always use `@functools.wraps(func)` inside a decorator so the wrapped
function keeps its `__name__`, `__doc__`, and `__wrapped__` attribute.
Without it, debugging and introspection get confused.
```

Prose outside section fences trains via continued pretraining;
instruction blocks (`### Q` / `### A`) train via SFT.

## 2. Run the training loop

```sh
$ uv run dlm train tutor.dlm
```

DLM runs the hardware doctor, resolves the plan (precision,
batch size, grad accumulation), downloads the base model (cached on
re-runs), and kicks off the SFTTrainer. On a Mac M-series with MPS,
20 steps of SmolLM2-135M take about two minutes.

Output (abbreviated):

```
preflight: 9.6 GB free under ~/.dlm/store/01KC…/
banner:    seed=42 determinism=best-effort plan=fp16/sdpa/bs=1×8
step 5:    loss=3.421  lr=5.00e-04
step 10:   loss=2.887  lr=4.47e-04
step 15:   loss=2.541  lr=3.45e-04
step 20:   loss=2.298  lr=2.08e-04
trained:   v0001 (20 steps, seed=42, determinism=best-effort)
adapter:   ~/.dlm/store/01KC…/adapter/versions/v0001
log:       ~/.dlm/store/01KC…/logs/train-000001-…jsonl
```

## 3. Inspect the store

```sh
$ uv run dlm show tutor.dlm
dlm_id:        01KC…
base_model:    smollm2-135m
training_runs: 1
    run 1 → v0001, 20 steps, seed=42, loss 2.30
adapter:       v0001
manifest:      ~/.dlm/store/01KC…/manifest.json
lock:          ~/.dlm/store/01KC…/dlm.lock
```

Under the hood, each run produced:

- `adapter/versions/v0001/adapter_config.json` + `adapter_model.safetensors` — the LoRA weights
- `adapter/versions/v0001/training_state.pt` + `.sha256` — optimizer/scheduler/RNG sidecar (for bit-exact resume)
- `manifest.json` — one `TrainingRunSummary` + the `content_hashes` delta
- `logs/train-000001-*.jsonl` — per-step metrics
- `dlm.lock` — pinned versions + hardware tier + determinism contract

## 4. Retrain after edits

Edit the document, add more Q&A pairs, then:

```sh
$ uv run dlm train tutor.dlm
```

The delta system (audit-04 M1/M2) compares `content_hashes` in the
manifest against the current sections, so only new content drives the
new training signal — everything from v0001 is still in the replay
corpus and gets sampled into the v0002 training mix.

Want to force a clean restart instead?

```sh
$ uv run dlm train tutor.dlm --fresh
```

## Next

You have a trained adapter. [Prompt it](first-prompt.md) next.
