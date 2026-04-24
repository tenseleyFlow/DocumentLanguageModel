# Preference section reference

`::preference::` sections are the pairwise alignment format DLM feeds
into the preference-training path (`dpo` / `orpo`). They are valid in
hand-authored `.dlm` files and in auto-mined output written by
`dlm preference mine --apply`.

## Basic shape

Each record contains three labeled blocks:

```dlm
::preference::
### Prompt
Explain recursion to a beginner.

### Chosen
Recursion is when a function calls itself on a smaller version of the
same problem.

### Rejected
Recursion is a self-referential computational strategy implemented with
stack-managed frame expansion.
```

One `::preference::` section can hold one or more Prompt/Chosen/Rejected
triples. DLM splits them into preference rows at parse time.

## Semantics

- `Prompt` is the input shown to the model.
- `Chosen` is the preferred response.
- `Rejected` is the lower-quality alternative.

Preference training does not try to predict the `Rejected` text.
Instead, it learns to increase the model's relative preference for the
Chosen response over the Rejected one.

## Auto-mined sections

When `dlm preference mine` writes sections back into a document, it
marks them with an HTML comment immediately after the section fence:

```dlm
::preference::
<!-- dlm-auto-mined: judge_name="sway" judge_score_chosen="0.82" judge_score_rejected="0.31" mined_at="2026-04-23T18:42:11Z" mined_run_id="7" -->
### Prompt
What is 2 + 2?
### Chosen
4.
### Rejected
The sum of two and two is four.
```

That marker corresponds to these parsed fields on the section:

- `auto_mined: true`
- `judge_name`
- `judge_score_chosen`
- `judge_score_rejected`
- `mined_at`
- `mined_run_id`

These metadata fields are required together for auto-mined preference
sections. Hand-authored sections omit the marker and keep
`auto_mined=false`.

## Validation rules

- The auto-mined marker is only valid on `::preference::` sections.
- Auto-mined sections must provide all metadata fields together.
- The parser rejects malformed score/timestamp/run-id values rather than
  silently guessing.
- Section identity ignores the auto-mined metadata, so the same logical
  preference pair keeps the same content identity whether it was written
  by hand or mined automatically.

## Interaction with training

- `dlm train` includes auto-mined preference sections by default.
- `dlm train --no-mined` excludes only `auto_mined=true` sections and
  still uses hand-authored preference pairs.
- Replay snapshots also preserve the `auto_mined` bit so future
  preference runs can opt in or out consistently.

## Related commands

- `dlm preference mine <path>`
- `dlm preference apply <path>`
- `dlm preference revert <path>`
- `dlm train <path> --no-mined`

## See also

- [Section grammar](sections.md)
- [CLI reference](../cli/reference.md)
- [Self-improving loop cookbook](../cookbook/self-improving-loop.md)
