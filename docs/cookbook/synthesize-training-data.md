# Synthesize training data

`dlm synth instructions` turns prose-heavy `.dlm` files into usable
`::instruction::` sections.

This is the shortest path from "I have notes" to "I have supervised
training pairs" when the document already contains domain prose but not
enough authored Q/A.

## What it does

The synth loop:

1. Finds non-empty prose sections in the document.
2. Prompts a teacher model to generate question/answer pairs about that
   prose.
3. Deduplicates the generated pairs.
4. Optionally filters them through the `sway` judge.
5. Either stages the accepted `auto_synth` sections for inspection or
   writes them straight back into the `.dlm`.

The generated sections are still normal `::instruction::` sections.
They just carry provenance metadata so DLM can tell synthesized pairs
from hand-authored ones.

## Choose a teacher

The teacher decides who writes the candidate Q/A pairs:

- `self`: use the current local adapter for this document
- `hf:<model>`: use a HuggingFace text model
- `openai:<model>`: use the OpenAI API
- `anthropic:<model>`: use the Anthropic API
- `vllm-server:<url>`: use an OpenAI-compatible local server

The current default is `self`, but that only makes sense once the
document already has a trained adapter. For a cold start, either:

- train once first, then synth with `self`, or
- use `hf:` / `openai:` / `anthropic:` / `vllm-server:` as the teacher

## Minimal example

Start with a prose-heavy document:

```dlm
---
dlm_id: 01K...
dlm_version: 15
base_model: smollm2-135m
---

DGEMM multiplies two dense matrices and can optionally accumulate the
result into an existing output matrix.
```

Generate one extraction-style pair per prose section with an HF teacher:

```sh
uv run dlm synth instructions notes.dlm \
  --teacher hf:Qwen/Qwen2.5-1.5B-Instruct \
  --per-section 1 \
  --strategy extraction
```

That prints two summaries:

- the raw synth plan
- the filter report (`generated`, `dedup`, `judge passed`, `threshold`)

By default, accepted sections are staged under the store so you can
inspect them:

```sh
uv run dlm synth list notes.dlm
```

If you want the accepted pairs written straight back into the document,
use `--apply`:

```sh
uv run dlm synth instructions notes.dlm \
  --teacher hf:Qwen/Qwen2.5-1.5B-Instruct \
  --per-section 1 \
  --strategy extraction \
  --apply
```

## Strategy choices

The `--strategy` flag controls what kind of questions the teacher is
asked to produce:

- `extraction`: questions answered directly by the prose
- `expansion`: questions a curious reader might ask beyond the exact
  wording of the prose
- `both`: split the per-section budget across both prompt styles

Start with `extraction` when you care about faithfulness. Reach for
`expansion` once the document already has a stable domain voice and you
want broader instructional coverage.

## Filter choices

The `--filter` flag controls post-generation cleanup:

- `sway`: dedup plus judge filtering against an empty baseline
- `dedup-only`: keep only near-duplicate suppression
- `none`: accept everything that parses as a valid pair

`sway` is the safest default and is what most users should keep. It is
especially helpful when using creative teachers or `--strategy both`.

If you are debugging prompt quality, use `--filter none` once and look
at the raw plan before deciding whether the issue is generation or
filtering.

## Useful knobs

```sh
uv run dlm synth instructions notes.dlm \
  --teacher hf:Qwen/Qwen2.5-1.5B-Instruct \
  --per-section 3 \
  --strategy both \
  --filter sway \
  --threshold 0.2 \
  --max-pairs 8 \
  --max-new-tokens 512 \
  --temp 0.2 \
  --top-p 0.95 \
  --seed 7
```

The most useful flags in practice are:

- `--per-section`: generate more than one candidate pair per prose block
- `--max-pairs`: cap document churn on large files
- `--threshold`: tighten or loosen `sway` acceptance
- `--temp` and `--top-p`: increase diversity when the teacher is too
  repetitive

## Training after synth

Once the document has accepted `auto_synth` instruction sections, the
next normal train run consumes them like any other instruction pair:

```sh
uv run dlm train notes.dlm
```

No special train flag is needed. Synthesized instruction sections flow
through the same SFT path as hand-authored sections.

## Revert and inspection

List applied auto-synth sections:

```sh
uv run dlm synth list notes.dlm
```

Strip every synthesized instruction section from the document:

```sh
uv run dlm synth revert notes.dlm
```

This only removes `auto_synth: true` instruction sections. Hand-authored
instruction blocks stay untouched.

## Common failure modes

### The self teacher is weak

If `--teacher self` produces junk, the adapter probably is not ready
yet. Train once more first, or use a stronger external teacher for the
first synth pass.

### Everything gets filtered out

That usually means one of three things:

- the teacher produced near-duplicates
- the generated answers were worse than the empty-baseline comparison in
  `sway`
- the threshold is too strict

Lower `--threshold`, or temporarily switch to `--filter dedup-only` to
see whether the judge is the main bottleneck.

### The document churns too much

Use `--max-pairs` aggressively at first. A small accepted batch is much
easier to reason about than dumping dozens of synthetic sections into a
single file.

## See also

- [Instruction section reference](../format/instruction-section.md)
- [Bootstrap self-improving](bootstrap-self-improving.md)
- [Self-improving loop](self-improving-loop.md)
- [CLI reference](../cli/reference.md)
