# Instruction section reference

`::instruction::` sections are the supervised fine-tuning format DLM
uses for prompt/answer training data.

They are valid in hand-authored `.dlm` files and in synthetic output
written by `dlm synth instructions --apply`.

## Basic shape

Each instruction section contains one or more `Q` / `A` pairs:

```dlm
::instruction::
### Q
What is a decorator?

### A
A function that takes a function and returns a wrapped function.

### Q
When should I use `functools.wraps`?

### A
Whenever a decorator returns another callable and you want to preserve
the wrapped function's metadata.
```

DLM splits those into individual supervised rows at parse time.

## Semantics

- `Q` is the prompt shown to the model.
- `A` is the target response.

At train time, DLM uses the question as context and the answer as the
supervised target. This is the section type that most directly shapes
assistant behavior.

## Auto-synth instruction sections

When `dlm synth instructions` writes sections back into a document, it
adds an HTML marker immediately after the section fence:

```dlm
::instruction::
<!-- dlm-auto-synth: synth_teacher="self" synth_strategy="extraction" synth_at="2026-04-24T10:18:42Z" source_section_id="b6b7d8a2f4b3f9c0" -->
### Q
What does DGEMM do?

### A
It multiplies dense matrices and can optionally accumulate the result.
```

That marker corresponds to these parsed fields on the section:

- `auto_synth: true`
- `synth_teacher`
- `synth_strategy`
- `synth_at`
- `source_section_id`

Hand-authored instruction sections omit the marker and keep
`auto_synth=false`.

## Validation rules

- The auto-synth marker is only valid on `::instruction::` sections.
- Auto-synth sections must provide all metadata fields together.
- `synth_teacher` and `synth_strategy` must be non-empty strings.
- `source_section_id` must be a valid referenced section ID.
- Section identity ignores the synth metadata, so the same logical
  question/answer pair keeps the same content identity whether it was
  written by hand or synthesized automatically.

## Interaction with training

- `dlm train` includes synthesized instruction sections by default.
- There is currently no separate "ignore auto-synth instructions" train
  flag; they flow through the normal SFT path once they are present in
  the document.
- `dlm synth revert` strips every `auto_synth: true` instruction section
  from the file without touching hand-authored rows.

## Interaction with `dlm synth`

Relevant commands:

- `dlm synth instructions <path>`
- `dlm synth list <path>`
- `dlm synth revert <path>`

The current `instructions` command can:

- stage accepted synth sections for inspection
- write accepted synth sections directly with `--apply`
- preview only with `--dry-run`

## Choosing a good instruction section

Hand-authored or synthesized, good instruction sections tend to have:

- a clear prompt with one task
- an answer that matches the tone you want the adapter to learn
- enough domain specificity that the pair teaches something real

Weak instruction sections tend to be:

- generic
- repetitive
- too broad to answer well
- stylistically inconsistent with the rest of the document

## See also

- [Section grammar](sections.md)
- [Synthesize training data](../cookbook/synthesize-training-data.md)
- [Bootstrap self-improving](../cookbook/bootstrap-self-improving.md)
- [CLI reference](../cli/reference.md)
