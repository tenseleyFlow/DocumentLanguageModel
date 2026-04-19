# Section grammar

Everything after the closing `---` of the frontmatter is the document
body. DLM's body parser splits it into typed **sections** using fence
markers of the form `::<type>::` on a line by themselves.

## Section types

### Prose (default)

Any body text that isn't inside an explicit fence is a prose section.
Prose trains via **continued pretraining** — the model learns the
writing style + vocabulary but doesn't get "question → answer" pressure.

```dlm
# Heading

Prose paragraphs, markdown code blocks, whatever you'd normally write.

Another paragraph after a blank line stays in the same prose section.
```

Code fences (` ``` `) inside prose are preserved; the parser doesn't
interpret `::type::` lines that appear inside a code block.

### Instruction (`::instruction::`)

Open with `::instruction::` on its own line. Each Q&A pair uses
`### Q` and `### A` as grammar markers.

```dlm
::instruction::
### Q
What is a decorator?

### A
A function that takes a function and returns a new function.

### Q
When should I use functools.wraps?

### A
Always, inside decorators.
```

Trains via **supervised fine-tuning (SFT)**: the model sees `Q` text
as the prompt, `A` text as the target. This is the pattern that
produces "helpful assistant" behavior.

### Preference (`::preference::`)

Open with `::preference::`. Each record has three blocks:

```dlm
::preference::
### Prompt
Explain recursion to a beginner.

### Chosen
Recursion is when a function calls itself on a smaller piece of the
problem. Imagine matryoshka dolls.

### Rejected
A recursive function is any function that refers to itself in its own
definition using the stack frame protocol.
```

Trains via **DPO** (direct preference optimization) or **ORPO** — the
model learns to prefer the `Chosen` phrasing. The DPO / ORPO trainer
lands in Sprint 17/18.

## Fence rules

- A fence must be the full line — `::instruction::` with no leading/
  trailing content other than whitespace.
- Fences inside triple-backtick code blocks are **not** active — the
  parser is aware of the code-fence context.
- An unfenced heading (`# ...`, `## ...`) inside an open instruction or
  preference section does **not** close the section. Close with the
  next section fence or end-of-file.
- Section type is case-sensitive; `::Instruction::` is rejected.
- Sprint 20 introduces a `::type#adapter-name::` suffix for
  multi-adapter routing; the v1 parser accepts the suffix but ignores
  the `#...` tail.

## Section IDs

Every section gets a content-addressed ID — the first 16 hex chars of
the SHA-256 of the section's canonical text. The manifest's
`content_hashes` records these IDs and their types so the next `dlm train`
can compute what's new, unchanged, or removed (Sprint 08's delta system).

You don't write these IDs in the document — they're derived and live
only in the manifest. But if you're debugging "why isn't this section
being picked up as new?", the ID in `dlm show --json` is the answer.

## What NOT to put in sections

- API keys, personal data, anything you wouldn't want baked into a
  model you'll share. The adapter learns from everything in the file.
- JSON / YAML config that the model should emit literally — use
  instruction Q&A pairs instead. Training on raw config produces
  noisy generation.
- Massive code dumps (>200 KB). The replay corpus retains everything,
  and sequence_len is bounded at 32 KB; a single enormous section
  trains one step and wastes the remaining token budget.

## See also

- [First train walkthrough](../getting-started/first-train.md)
- [Cookbook: coding tutor](../cookbook/coding-tutor.md) — full
  example of instruction-heavy authoring
