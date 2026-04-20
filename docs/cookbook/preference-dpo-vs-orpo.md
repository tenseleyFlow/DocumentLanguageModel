# Preference tuning: DPO vs ORPO

Once your document has `::preference::` sections, you get a choice of
alignment method: **DPO** (Direct Preference Optimization) or **ORPO**
(Odds-Ratio Preference Optimization). Both consume the same
`### Prompt / ### Chosen / ### Rejected` triples; they differ in how
they turn those triples into gradient updates.

This page is the cheat sheet for picking.

## The `::preference::` section

```dlm
::preference::
### Prompt
What is 2 + 2?
### Chosen
4.
### Rejected
The sum of two and two is four, a basic arithmetic fact.

### Prompt
What color is grass?
### Chosen
Green.
### Rejected
Grass is typically a vibrant shade of green most of the year.
```

Write as many triples as you want per section; `::preference::` blocks
stack just like `::instruction::` ones.

## Picking a method

```yaml
training:
  preference:
    method: dpo   # or orpo
    hyperparams:
      beta: 0.1      # DPO's KL weight
      alpha: 0.1     # ORPO's odds-ratio weight
      learning_rate: 5e-6
      num_epochs: 1
    loss_type: sigmoid   # DPO only
    reference: pre_adapter  # DPO only
```

`method` defaults to `dpo`. Omit the block entirely to disable
preference training; it flips on automatically as soon as you add a
`::preference::` section.

## The tradeoff

| | DPO | ORPO |
|---|---|---|
| Reference model | required (base or SFT adapter) | none |
| Memory | ~2× the policy (policy + frozen reference) | ~1× |
| Hyperparameter sensitivity | high — beta is twitchy | moderate |
| Mixes SFT + preference | separate phase (SFT writes v_N, DPO writes v_{N+1}) | combined objective, still writes v_{N+1} |
| Convergence with small datasets | noisy; wants hundreds to thousands of pairs | stable on 5–50 pairs |
| Hardware plan | memory doubles; `dlm doctor` halves the micro-batch | same batch as SFT |

## Rule of thumb

- **Start with ORPO.** Fewer moving parts, lower memory, robust on
  the small preference sets a single document realistically contains.
- **Switch to DPO** when you have ≥1k preference pairs *and* the
  reference-anchored objective is what you want (e.g., you're
  deliberately nudging a chat model away from a known base).
- **Don't try to mix them in one run.** `method` is a switch, not a
  blend. If you've already run DPO, flipping to ORPO on the next
  `dlm train` just means the next adapter version comes out of ORPO;
  prior versions are untouched.

## Runnable: five terse-favoring pairs

```sh
# Create a .dlm with default base + starter instructions.
uv run dlm init tutor.dlm --base smollm2-135m --i-accept-license

# Edit: add a ::preference:: section with five (or more) triples that
# favor terse answers over verbose ones.

# First pass: SFT on whatever prose/instruction is there, then ORPO
# on the preferences, in one invocation.
uv run dlm train tutor.dlm

# The manifest now has two TrainingRunSummary entries:
uv run dlm show tutor.dlm
```

Run only the preference phase (e.g., to retune `alpha` without
redoing SFT):

```sh
uv run dlm train tutor.dlm --phase preference
```

Skip preferences entirely for a regression comparison:

```sh
uv run dlm train tutor.dlm --phase sft
```

## Migrating from v1

Documents authored before ORPO landed used `training.dpo`. The parser
refuses v1 docs in v2 CLIs, but `dlm migrate` rewrites them in place:

```sh
uv run dlm migrate tutor.dlm
# mydoc.dlm.bak keeps the pre-migration copy for rollback.
```

The migration renames `training.dpo` → `training.preference`, groups
`beta / learning_rate / num_epochs` under `hyperparams:`, and updates
`reference: pre_dpo_adapter` → `reference: pre_adapter`. The method
defaults to `dpo` so the migrated doc behaves identically to its v1
self.

## See also

- [First train walkthrough](../getting-started/first-train.md) for
  the single-phase flow this builds on
- [Determinism](../determinism.md) — the preference method and
  hyperparams participate in the `dlm.lock` reproducibility record
