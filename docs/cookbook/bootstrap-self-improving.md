# Bootstrap self-improving

The self-teacher loop is the most interesting version of Sprint 43:
your current adapter writes new `::instruction::` sections for its own
document, then the next train run folds them back in.

This is not magic. It works because DLM already has:

- replay-backed retraining
- synthesized instruction provenance (`auto_synth`)
- a local `sway` judge for filtering weak candidates

Used carefully, it turns one trained document into a steadily better
instruction corpus.

## The honest starting point

`--teacher self` uses the current adapter for that `.dlm`. That means
the loop starts **after** there is already a trainable local adapter.

A good bootstrap pattern is:

1. Start with prose plus at least some useful seed supervision, or do an
   initial train from prose and existing sections.
2. Run `dlm synth instructions --teacher self`.
3. Retrain on the accepted synth sections.
4. Repeat in small batches.

If the adapter still cannot answer basic questions about the document,
synthetic instruction generation will mostly amplify noise.

## Minimal loop

Train once:

```sh
uv run dlm train notes.dlm
```

Generate a small accepted batch from the current adapter and write it
back immediately:

```sh
uv run dlm synth instructions notes.dlm \
  --teacher self \
  --per-section 1 \
  --strategy extraction \
  --max-pairs 4 \
  --apply
```

Retrain on the expanded instruction set:

```sh
uv run dlm train notes.dlm
```

Then inspect real output quality:

```sh
uv run dlm prompt notes.dlm "What does DGEMM do?"
```

That is the basic self-improving loop.

## Safer staged version

If you want to inspect before writing:

```sh
uv run dlm synth instructions notes.dlm \
  --teacher self \
  --per-section 1 \
  --strategy extraction

uv run dlm synth list notes.dlm
```

The current implementation stages accepted synth sections for
inspection, but it does not yet have a separate `dlm synth apply`
subcommand. Use `--apply` on the synth run when you want the sections
written straight into the document.

## Why `sway` stays the default

The self-teacher path is the place where the default `--filter sway`
matters most.

Without filtering, a weak adapter can happily generate:

- duplicates
- overly generic answers
- plausible but wrong extrapolations

The current synth filter stack is:

1. dedup
2. optional judge pass
3. optional threshold cut

The CLI prints those counts so you can tell whether the loop is getting
better or just louder.

## A conservative rhythm

This is a healthy local rhythm for a real project:

```sh
uv run dlm train notes.dlm
uv run dlm synth instructions notes.dlm \
  --teacher self \
  --per-section 1 \
  --max-pairs 4 \
  --apply
uv run dlm train notes.dlm
uv run dlm prompt notes.dlm "Explain the core idea."
```

Keep the accepted batch small at first. The point is to improve the
document's instruction surface, not flood it with speculative rows.

## When to switch away from `self`

The self-teacher is convenient, but not always the right teacher.

Prefer an external teacher when:

- the local adapter is still very early and weak
- you need broader general knowledge than the current adapter can supply
- you want to compare local-vs-external synth quality on the same prose

That usually looks like:

```sh
uv run dlm synth instructions notes.dlm \
  --teacher hf:Qwen/Qwen2.5-1.5B-Instruct \
  --per-section 1 \
  --apply
```

and then later moving back to `--teacher self` once the adapter has real
domain traction.

## Pairing Sprint 43 with Sprint 42

Instruction synthesis and preference mining are complementary:

- `dlm synth instructions` grows the SFT side of the document
- `dlm synth preferences` / `dlm preference mine` sharpens ranking and
  behavior once the adapter can already produce multiple plausible
  answers

A practical sequence is:

1. train
2. synth instructions
3. train
4. mine preferences
5. train preference phase

That is the closest current DLM path to a fully local self-improving
document loop.

## Failure modes to watch

### The second pass is not better

That usually means one of:

- the first synth batch was too weak
- the document still lacks enough domain prose
- the adapter is too small for the domain

Do not assume "more synthetic rows" automatically means "better model."

### Expansion mode gets weird

`--strategy expansion` is useful, but it is also the fastest route to
polished nonsense. Prefer `extraction` for early loops and only widen to
`both` or `expansion` once the adapter is already grounded.

### Prompt quality improves but factuality does not

That is a signal to go back to better prose or hand-authored
instructional supervision. Self-improvement cannot invent missing source
knowledge.

## See also

- [Synthesize training data](synthesize-training-data.md)
- [Instruction section reference](../format/instruction-section.md)
- [Self-improving loop](self-improving-loop.md)
- [Reward-model integration](reward-model-integration.md)
