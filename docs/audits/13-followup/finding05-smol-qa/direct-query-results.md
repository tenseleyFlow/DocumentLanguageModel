# Finding 05 — direct query transcripts

**Setup:** SmolLM2-135M, 35 INSTRUCTION pairs, r=16/α=32/lr=2e-4. Trained
60 epochs (early-stop fired). Final train loss 1.45 instantaneous 0.31,
**eval_loss 2.37**, eval_acc 0.63. Compare finding 04 same recipe on
qwen-coder-1.5b: eval_loss **0.047**, eval_acc 0.986.

Adapter: `~/.dlm/store/01KQGBY1QWPFBKE0E8487PRB3E/adapter/versions/v0001`
Backend: `--backend pytorch` (chat-templated).

## Q1 — wedge: stdlib sorting signature (trained verbatim)

> Show the signature of stdlib's sorting routine in Fortran.

```
```fortran
use stdlib_sort, only: sort
call sort(array[, reverse])
```
The `array` argument is `'list'` (ordered array of elements) and
`'string'` (same length as `array[]').ortunately, it preserves
key-value pairs.

The `array[]` is the argument's selling point. It returns a rank-1 array
of `array[]` whose elements must contain the same number of elements
(the sorted part of the `sort` Kubo algorithm).
```

**Verdict:** verbatim correct for the *first 3 lines* (call signature
matches training), then degenerates into hallucinated post-completion
("Kubo algorithm" doesn't exist; "selling point" is incoherent). Note
even the verbatim part is slightly broken: `stdlib_sort` should be
`stdlib_sorting`. Compare finding 04's clean reproduction.

## Q2 — trained module question

> What module in fortran-lang/stdlib provides sorting routines?

```
ord_sort and sort_index ::utiveaurconditions — although exch orders is
downgrade to ord_sort via order_stdap, they do not use it. also quad_sort does
not use radix order.
```

**Verdict:** memorized the right names (`ord_sort`, `sort_index`,
`quad_sort`, `radix`) but the connecting tissue is gibberish. The
LoRA pulls in trained vocabulary but can't reconstruct the surrounding
sentence structure. Compare finding 04 which produced the full
trained answer verbatim.

## Q3 — out-of-domain preservation: capital of France

> What is the capital of France?

```
A Fortified AI, spoiled by the NAM module! The city of Paris is the country's
most populous urban area. The region encompasses the
```

**Verdict:** **base capability damaged.** Pre-prefix is fortran-domain
bleed ("Fortified AI", "NAM module") that doesn't appear in the
training corpus — it's compositional overflow from the LoRA delta
into general English text. Finding 04 returned this query unchanged
from base ("Paris" with no domain bleed).

## Q4 — held-out same-module generalization

> How would I sort a real(dp) array using stdlib?

```
The `real_dp([kind])]` returns a kind ofrasse: big ends are
exactly `n` days away from the beginning, but theraels can
include* themselves. The `todap:`~stdlib_sorting` special_surrounding
is swapped with the `sort` call. The `sort` is instead done with
standard
```

**Verdict:** incoherent. `real_dp([kind])]`, "big ends ... days away",
"theraels" — token salad. Neither memorization nor generalization;
the model has lost the ability to write coherent text on this prompt
shape. Finding 04 was "knows `stdlib_sorting`, invents wrong call form" —
i.e. partial generalization. Smol shows zero generalization.

## Q5 — held-out different module

> How do you write a CSV file with stdlib?

```
```fortran
use stdlib_resources, only: numbers_and_text, lines:
  numbers :: line.integer(0) - integer, nargs-plus(length: 1)
    length :: integer(size)
    ! single_element
    loop(*) $$
    ? .. [loop]
    .. [single_element]
    .. [single_element] *length
```

**Verdict:** hallucinated `stdlib_resources` (not a real module),
malformed fortran (`nargs-plus`, `loop(*) $$`, `[single_element]`).
Finding 04 invented a plausible-looking but wrong API for similar
held-out questions; smol invents a non-plausible, non-syntactic mess.

## Q6 — trivial arithmetic: base preservation control

> What is 2 + 2?

```
Yes, you're calling the standard `stdlib_array_plus` helper function. `2
```

**Verdict:** **catastrophic preservation failure.** The model interprets
"2 + 2" as a fortran library question and hallucinates a `stdlib_array_plus`
helper. Base SmolLM2-135M-Instruct answers "4". This is the
finding-02 cal_general regression in concentrated form — the LoRA
hijacks general queries through training-domain priors.

## Summary

| Question type | Finding 04 (qwen-coder-1.5b) | Finding 05 (smol) |
|---|---|---|
| **Trained verbatim** | clean reproduction | partial verbatim then gibberish |
| **Held-out same-module** | partial generalization | token salad |
| **Held-out different module** | plausible-looking wrong API | non-syntactic hallucination |
| **Out-of-domain** ("capital of France") | unchanged from base | fortran bleed-through |
| **Trivial arithmetic** ("2 + 2") | (not tested, likely correct) | hallucinates `stdlib_array_plus` |

The recipe-shape fix that worked on qwen-coder-1.5b doesn't rescue
smol. The architectural floor diagnosed in finding 02 reasserts itself
even with the cleanest possible training-data shape: at 135M params,
the LoRA can't compose a learned domain with general chat capability —
adding the domain *destroys* the chat capability.
