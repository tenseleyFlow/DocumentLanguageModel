# Reward-model integration

`dlm preference mine` can score candidate answers with something other
than the adapter itself.

That is the point of the judge selector:

```sh
uv run dlm preference mine mydoc.dlm --judge sway
uv run dlm preference mine mydoc.dlm --judge hf:YourOrg/reward-model
uv run dlm preference mine mydoc.dlm --judge 'cli:/path/to/judge-bin'
```

This page is the practical guide for the two non-default paths:
HuggingFace reward models and external CLI judges.

## Why use a reward model at all

The default `sway` judge is a bootstrap convenience. It is fast to reach
for, but it is still the adapter judging its own candidates.

Use an external judge when:

- the adapter is still small or early in training
- you care about policy or style adherence more than raw task accuracy
- you already have a reward model or scoring binary your team trusts

## HuggingFace reward models

Point `--judge` at a sequence-classification model:

```sh
uv run dlm preference mine mydoc.dlm \
  --judge hf:OpenAssistant/reward-model-deberta-v3-large-v2 \
  --threshold 1.0 \
  --samples 4 \
  --max-pairs 10
```

DLM loads the model lazily, scores each candidate pair, and keeps only
those whose chosen-vs-rejected margin clears the threshold.

### Thresholds

The default threshold depends on the judge implementation:

- `sway`: `0.1`
- `hf:<model>`: `1.0`

Raise the threshold when you want fewer, higher-confidence pairs. Lower
it when the judge is too conservative and you are getting almost no
output.

## External CLI judges

The `cli:` path is for custom scorers, policy engines, or internal
reward-model wrappers.

Example:

```sh
uv run dlm preference mine mydoc.dlm \
  --judge 'cli:/usr/local/bin/rank-answer-pair' \
  --samples 4
```

The judge process is invoked once per candidate. It receives JSON on
stdin and must answer with JSON on stdout.

Input shape:

```json
{
  "prompt": "What is DGEMM?",
  "candidate": "A matrix multiply."
}
```

Output shape:

```json
{
  "score": 0.9,
  "reasoning": "Specific, correct, and terse."
}
```

If the command cannot be invoked or emits malformed JSON, the mine run
fails fast instead of silently accepting garbage scores.

## A good reward-model workflow

Start small and observable:

```sh
uv run dlm train mydoc.dlm
uv run dlm preference mine mydoc.dlm \
  --judge hf:YourOrg/reward-model \
  --samples 4 \
  --max-pairs 6
uv run dlm preference list mydoc.dlm
uv run dlm preference apply mydoc.dlm
uv run dlm train mydoc.dlm --phase preference
```

Then inspect:

```sh
uv run dlm metrics show mydoc.dlm --run-id 7 --json
uv run dlm prompt mydoc.dlm "..." 
```

Judge-score improvement is not enough on its own. Always check held-out
behavior from the adapter you just trained.

## Common mistakes

### Using reward mining for missing knowledge

Reward models pick between candidate answers. They do not invent facts
the base adapter never learned. If the model is simply wrong, go back to
SFT data first.

### Mining too many pairs too early

If the reward model is stronger than the adapter, it can still rank a
batch of uniformly weak answers. Cap with `--max-pairs` and inspect the
result before turning it into a habit.

### Trusting only the reward score

Repeated reward-driven loops can drift into reward hacking. Watch actual
task outputs, not just margins.

## When `sway` is still enough

Stay with the default judge when:

- you are iterating locally on tone or terseness
- the document is small and you want the lowest-friction loop
- you mainly need a filter for "better vs worse" candidates, not a
  strong external policy model

Move to `hf:` or `cli:` when the loop starts to matter to other people.

## See also

- [Self-improving loop](self-improving-loop.md)
- [Preference tuning: DPO vs ORPO](preference-dpo-vs-orpo.md)
