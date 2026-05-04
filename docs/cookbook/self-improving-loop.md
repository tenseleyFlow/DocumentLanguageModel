# Self-improving loop

`dlm preference mine` closes the gap between "I have an adapter" and
"I have new preference pairs to train on."

The loop is simple:

1. Train an initial adapter from prose and `::instruction::` sections.
2. Mine auto-ranked `::preference::` sections from that adapter.
3. Apply the mined sections back into the document.
4. Train the preference phase again.

This is the shortest honest path to "train once, judge outputs, train
again" without leaving the `.dlm`.

## When this works well

- You already have useful `::instruction::` prompts in the document.
- The adapter is good enough to generate multiple distinct answers.
- You want to sharpen style, brevity, refusal behavior, or task
  preference, not inject brand-new knowledge.

If the model is still too weak to produce meaningful alternatives, do
another SFT pass first. Preference mining is an alignment loop, not a
replacement for basic competence.

## Minimal loop

Start with a normal document that has at least one instruction section:

```dlm
::instruction::
### Q
How should release notes read?
### A
Short, factual, and low-drama.
```

Train once:

```sh
uv run dlm train release-notes.dlm
```

Mine a small batch of candidate pairs and write them straight into the
document:

```sh
uv run dlm preference mine release-notes.dlm \
  --samples 4 \
  --max-pairs 8 \
  --apply
```

Then train just the preference phase:

```sh
uv run dlm train release-notes.dlm --phase preference
```

That writes the next adapter version using the newly mined
`::preference::` sections.

## Safer first pass

If you want a review step before touching the document, omit `--apply`:

```sh
uv run dlm preference mine release-notes.dlm --samples 4 --max-pairs 8
uv run dlm preference list release-notes.dlm
uv run dlm preference apply release-notes.dlm
```

This stages the mined plan under the store, lets you inspect it, and
only then writes the sections into the `.dlm`.

## What gets written

Auto-mined sections are still normal `::preference::` sections, but they
carry provenance fields:

- `auto_mined: true`
- `judge_name`
- `judge_score_chosen`
- `judge_score_rejected`
- `mined_at`
- `mined_run_id`

That means the next `dlm train` consumes them through the same
preference data path as hand-authored pairs.

## Using `--no-mined`

For A/B checks, keep the mined sections in the document but exclude them
from the preference phase:

```sh
uv run dlm train release-notes.dlm --phase preference --no-mined
```

This is useful when you want to compare:

- hand-authored preferences only
- mined + hand-authored preferences together

without deleting anything from the file.

## Observability

Use these two commands to see what happened:

```sh
uv run dlm metrics show release-notes.dlm --run-id 7 --json
uv run dlm show release-notes.dlm --json
```

`dlm metrics` surfaces per-run preference-mining events, including mined
pair counts and skipped prompts. `dlm show --json` adds the latest
preference-mining summary to the store snapshot.

## Picking a judge

The default judge is `sway`, which bootstraps from the current adapter.
That is convenient, but not always the best production choice.

- Use `sway` for quick local iteration and loop-shaping.
- Use `hf:<model>` when you already trust a reward model for the task.
- Use `cli:<cmd>` when your org has an external scorer or policy
  checker.

For the judge contract and thresholds, see
[Reward-model integration](reward-model-integration.md).

## Failure modes to watch

- Near-identical generations: raise `--temp`, or lower `--top-p`
  constraints so the sampler can explore.
- Weak base adapter: mine after another SFT pass, not before.
- Reward hacking: track held-out eval behavior, not just judge scores.
- Low-quality bootstrap self-judging: use an HF reward model on smaller
  bases instead of trusting `sway` alone.

## A concrete rhythm

This is a sane lightweight loop for a personal project:

```sh
uv run dlm train notes.dlm
uv run dlm preference mine notes.dlm --samples 4 --max-pairs 6 --apply
uv run dlm train notes.dlm --phase preference
uv run dlm prompt notes.dlm "Write this week's changelog intro."
```

Run that loop when the adapter's behavior is close but still annoying.
Do not run it just to accumulate pairs for their own sake.

## See also

- [Preference tuning: DPO vs ORPO](preference-dpo-vs-orpo.md)
- [Reward-model integration](reward-model-integration.md)
- [Metrics & observability](metrics.md)
