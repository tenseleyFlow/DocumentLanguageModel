# Tag-weighted training corpus

When you train across a codebase, some files deserve more attention
than others. Handwritten docstrings teach tone. Generated code teaches
conventions you'd rather forget. The tag-weighted corpus knob lets you
declare that preference **in the codebase itself** via
`.dlm/training.yaml`, not in the `.dlm` frontmatter — so the weighting
travels with the code.

## Shape

```yaml
# ~/code/my-repo/.dlm/training.yaml
dlm_training_version: 1

metadata:
  lang: python

weights:
  lang:
    python: 1.0
    generated: 0.1
  docstring:
    "true": 2.0
```

`metadata` (Sprint 30) tags every file ingested under this anchor.
`weights` (Sprint 36.1) then scales each row's exposure during
training:

- `weight > 1`: row appears more often (integer weight = N duplicate
  copies; fractional = deterministic additional copy with that
  probability seeded by `(training.seed, section_id)`).
- `weight < 1`: row appears with probability equal to the weight.
- `weight = 0`: row dropped entirely.
- Tags not declared in `weights`: unchanged (weight 1).

Multiple matching tags **multiply**: a row tagged
`{lang: python, docstring: "true"}` under the config above ends up
at `1.0 × 2.0 = 2.0` — two copies.

## Why row repetition, not loss scaling?

Implementing "give this row 2× attention" by multiplying its loss
sounds cleaner than duplicating it, but it would require subclassing
TRL's `SFTTrainer.compute_loss` — which rot quickly across TRL
versions. Row repetition is a **dataset-level transform**: every
downstream layer (pretokenize cache, TRL collator, AdamW, determinism
golden) sees a plain list of rows and stays dumb. The Sprint 31.5
bit-identity guarantee carries through unchanged.

Integer weights are mathematically equivalent to loss scaling under
SGD/AdamW — E[grad] = Σ wᵢ · gradᵢ / Σ wᵢ = (repeated rows) / N.
Fractional weights are approximate but stable; the deterministic
seeding keeps them byte-identical across runs.

## Nearest-ancestor merge

If you drop `.dlm/training.yaml` at multiple depths, the deepest
`(tag_key, tag_value)` entry wins — the same semantics
`.dlm/training.yaml`'s `metadata` and `exclude` already use:

```
~/code/my-repo/.dlm/training.yaml          # root: weights.lang.python = 1.0
~/code/my-repo/tests/.dlm/training.yaml    # subtree: weights.lang.python = 0.5
```

Under `tests/`, python files score 0.5×. Everywhere else, 1.0×.

## Worked example — fortran + generated code

Say your Fortran repo has hand-tuned solvers you want the model to
learn well, plus machine-generated Fortran from a preprocessor that's
mostly noise. Sprint 30's metadata tagging is the first half:

```yaml
# ~/FortranGoingOnForty/fortsh/.dlm/training.yaml
dlm_training_version: 1
metadata:
  lang: fortran
  domain: numerical
```

```yaml
# ~/FortranGoingOnForty/fortsh/generated/.dlm/training.yaml
dlm_training_version: 1
metadata:
  lang: fortran
  generated: "true"
```

Now add the weights at the root:

```yaml
# ~/FortranGoingOnForty/fortsh/.dlm/training.yaml (appended)
weights:
  generated:
    "true": 0.1
  domain:
    numerical: 1.5
```

Rows from `generated/` get 10% exposure; domain-tagged rows (every
file under the root anchor) get 1.5× exposure. The overall shape:
solvers learn well, generated noise doesn't drown them out.

## Auditing the expansion

After `dlm train`, the per-tag row counts land on the training run
summary:

```bash
dlm show /path/to/doc.dlm --json | jq '.manifest.training_runs[-1].weight_distribution'
# {
#   "lang": {"fortran": 847},
#   "generated": {"true": 312},
#   "domain": {"numerical": 847}
# }
```

This is the **pre-expansion** count — 847 Fortran rows, 312 of which
are generated. After expansion at the weights above:

- Non-generated rows: 535 rows × 1.5 = ~803 copies
- Generated rows: 312 rows × 0.1 × 1.5 = ~47 copies

A `null` `weight_distribution` means no `.dlm/training.yaml` in the
descent declared a `weights` block — the corpus went through
untouched.

## Edge cases

- **Weight 0 drops the row.** Use this to exclude entire classes of
  files without editing `exclude` globs.
- **Negative weights are rejected** at parse time — they have no
  well-defined meaning under row repetition.
- **No tags → weight 1.** Rows from in-body `::instruction::` or
  `::preference::` sections, or from directive paths that don't sit
  under a tagged subtree, are unaffected.
- **Determinism.** Same seed + same corpus → same expanded row list,
  bit-exact. Changing `seed` reshuffles fractional keep/drop
  decisions; integer parts are unaffected.
- **Interaction with replay.** Replay rows from the corpus are
  expanded too — they've got the same tag metadata from their
  originating training cycle. This keeps retention uniform.

## Related

- `docs/format/dlm-training-yaml.md` — the full schema reference
  including `metadata`, `include`, `exclude`, `exclude_defaults`.
- `docs/cookbook/training-across-codebases.md` — how `.dlm/`
  discovery feeds into training.
- `docs/cookbook/directive-cache.md` — tokenized-section cache
  interaction (expanded rows that share a `section_id` share a cache
  entry, so repetition is cache-free).
