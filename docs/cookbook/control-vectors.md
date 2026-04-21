# Control vectors

A **control vector** is a one-shot steering direction extracted from
`::preference::` sections. Unlike a LoRA adapter — which takes hours
of training to learn a preference — a control vector is computed
gradient-free in seconds, stored as a single small tensor, and
applied at inference via a forward-time hook on the residual
stream.

Use it when you want to steer *style* rather than *capability*:
formality vs. casualness, verbosity vs. concision, cautious vs.
direct. Capability work (teaching new facts, fixing bugs in code)
still wants a LoRA. Control vectors are orthogonal — you can stack
them over an already-trained adapter at inference time.

## The shape

Extraction reads `N` preference pairs. For each pair, the base
model is run on the `chosen` and `rejected` completions and
hidden states are captured at a residual-stream layer. The
difference `chosen_i - rejected_i` is a "pull toward chosen"
vector for that example. The first right-singular vector of the
stack of differences is the direction these pulls agree on —
that's the steering vector.

Applied at inference with strength `s`, the vector is added to
every token's hidden state at that layer during the forward pass:

```
hidden_state[t] += s * control_vector
```

Positive `s` pushes toward the `chosen` distribution; negative
pushes away. Typical range: `[-2, 2]`. Beyond `±3` the model
collapses into repetition.

## Workflow

### 1. Write a `::preference::` section

Pairs should isolate the *single dimension* you want to steer.
For formality, vary formality; keep topic and length constant.

```markdown
---
dlm_id: 01KP...
base_model: smollm2-135m
---

::preference#formal::
### Prompt
Explain what a mutex is.

### Chosen
A mutex (mutual exclusion lock) is a synchronization primitive
that ensures only one thread can access a shared resource at a
time. Threads that attempt to acquire a held mutex block until it
is released.

### Rejected
so basically a mutex is like a lock that makes sure two threads
don't trip over each other when they need the same thing. you grab
it, do your thing, let it go.
```

Add ~10-30 pairs for a usable direction. Fewer than 5 and the
signal is too noisy; more than 50 and you're past diminishing
returns.

### 2. Extract

With hidden states collected from the base model:

```python
import numpy as np
from dlm.control import extract_control_vector, refuse_if_policy_safety

# Validate that no preference section is tagged `policy: safety`.
refuse_if_policy_safety([section.tags for section in preference_sections])

# hidden_chosen, hidden_rejected: each (N, hidden_dim) arrays of
# residual-stream activations at the chosen layer.
vec = extract_control_vector(hidden_chosen, hidden_rejected)

print(f"n_pairs={vec.n_pairs}, explained_variance={vec.explained_variance:.2f}")
# n_pairs=20, explained_variance=0.73
#
# 0.73 = the principal component captures 73% of the total signal
# energy. Above ~0.5 is a coherent direction. Below ~0.3, the
# pairs are probably too noisy or contradictory — add more, or
# tighten the prompt template.
```

### 3. Persist

The per-store layout at `~/.dlm/store/<dlm_id>/controls/`:

```
controls/
    formal.safetensors     # the direction tensor
    formal.meta.json       # {layer_index, source_section_ids, n_pairs, extractor_version}
```

The meta JSON is how `dlm show` audits what produced a given
vector — source sections, layer, pair count, extractor version
(so future API changes can invalidate stale vectors deterministically).

### 4. Apply at inference

```python
from dlm.control import apply_control

with apply_control(model, vec.direction, layer_index=12, strength=1.5):
    out = model.generate(input_ids, max_new_tokens=128)
```

The hook attaches on `__enter__`, removes on `__exit__` — even if
the wrapped block raises. Leaving a hook active would silently
steer unrelated generations, so the context manager pattern is
load-bearing.

## Layer choice

`layer_index` picks which residual stream gets the perturbation.
Rules of thumb (Panickssery et al., 2024):

- **Middle layers** (40–60% depth) are the sweet spot for most
  style dimensions — formality, tone, caution.
- **Early layers** (0–20% depth) steer vocabulary and syntax but
  don't propagate cleanly through downstream composition.
- **Late layers** (80–100% depth) can change a few output tokens
  but leave the underlying reasoning unchanged.

For a 32-layer model, start at `layer_index=16`. Sweep `[8, 16,
24]` on a held-out prompt if the initial result is weak.

## Safety refusal

Preference sections tagged `policy: safety` are **refused at
extraction time**:

```markdown
::preference#safe-refuse::
tags:
  policy: safety
### Prompt
...
### Chosen
<safe refusal>
### Rejected
<unsafe compliance>
```

Extracting a vector from those pairs would produce a "more safety
vs less safety" direction — applied at negative strength, it
erodes the safety training the document is trying to preserve.
`refuse_if_policy_safety` surfaces the refusal before any
artifact reaches disk:

```
ControlPolicyRefusal: refusing to extract a control vector from
preference sections tagged `policy: safety` — the resulting
steering direction could be used at negative strength to undo
the safety training the document is trying to preserve.
```

The refusal is at extract time, not apply time, so the vector
never exists. Re-tagging to bypass the check is not supported;
the footgun is the shape of the math, not the tag.

## Validation

End-to-end sanity check for a newly-extracted vector:

1. Load the base model.
2. Generate without the vector on 20 held-out prompts.
3. Generate with `strength=1.0` on the same prompts.
4. Judge (LLM-as-judge or manual) whether the axis moved.

For a formality vector, you expect judge-formality-rating to
correlate positively with `strength`. A failed extraction looks
like: outputs identical at ±1, nonsense at ±2. That's the signal
to add more pairs, pick a different layer, or accept that the
dimension you're trying to steer isn't a linear subspace.

## What ships today

- `extract_control_vector` — raw-SVD over chosen/rejected
  differences, deterministic orientation (aligned with mean pull).
- `apply_control` — context-managed `forward_pre_hook` with
  shape + layer validation.
- `refuse_if_policy_safety` — pre-extraction safety gate.
- `ControlVector` dataclass with `n_pairs` + `explained_variance`
  for audit output.
- Per-store layout: `controls/<name>.safetensors` +
  `<name>.meta.json`.

## What's deferred

- **CLI surface** (`dlm control extract | apply | list`) — needs
  a real HF base model to drive the forward-pass residual
  collection. Land as a follow-up when the `dlm.inference.loader`
  integration is wired.
- **Multi-control composition** — additive for compatible layers,
  warn on conflicts. Single-control is the v1 shape.
- **Serialization format** — today the spec says safetensors.
  Landing safetensors I/O alongside the CLI keeps the two
  commits paired.
- **Integration with `dlm prompt`** — `--control name:strength[,...]`
  flag for the existing prompt path.

These are all layer-cake work on top of the extraction + apply
primitives shipped here; the math path is the hard-to-get-right
piece and it's done.

## Risks

- **Small bases are unstable.** Control vectors below ~500M
  parameters tend to collapse into repetition past `|strength| > 1`.
  `dlm doctor` will warn on bases below that threshold when the
  CLI lands.
- **Layer choice matters more than strength.** A wrong layer at
  strength 1 is worse than any strength at the right layer.
- **Control vectors are not a safety mechanism.** They're a
  steering *tool*. The `policy: safety` refusal is a footgun
  guard, not a security boundary — anyone who can train LoRAs
  on the same base can produce the same direction by other
  means. The safety concern is specifically about documents
  undoing their own safety training, not about external
  attackers.
