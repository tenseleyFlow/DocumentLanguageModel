# Finding 03 — base capability snapshot (pre-training)

`Qwen/Qwen2.5-Coder-1.5B-Instruct`, fp16, MPS, greedy decoding, no
adapter.

## Probe set

| # | Question | Verdict | Notes |
| --- | --- | --- | --- |
| 1 | How do you declare an allocatable array of `real(real64)` in modern Fortran? | **correct** | Idiomatic syntax, complete `program` block. |
| 2 | When should you prefer `do concurrent` over plain `do`? | partial | Mentions performance/parallelism but vague on loop-carried-dependence semantics. |
| 3 | Show the signature of stdlib's sorting routine in Fortran. | **WRONG** | Claims Fortran has no built-in sort, invents `iso_fortran_env::sort` which does not exist. |
| 4 | What does `intent(in)` mean for a dummy argument? | **correct** | Accurate explanation + example. |

## Baseline character

The base already speaks fortran fluently — syntax, modern features
(`do concurrent`, `intent`, `allocatable`), `program` structure. What
it does **not** know is **`fortran-lang/stdlib`** — the third-party
modular standard library at `/tmp/stdlib_build/src`. Q3 demonstrates
this directly: rather than pointing to `stdlib_sorting` or
`stdlib_sorting::sort`, the model fabricates a non-existent
`iso_fortran_env::sort`. This is exactly the kind of specific-corpus
knowledge a LoRA should be able to add.

## What this means for the experiment

The finding-03 hypothesis is now *narrow and falsifiable*:

> Training a LoRA on the audit-13-followup corpus (fortran source
> files + stdlib source + stdlib doc/specs markdown) should produce
> a measurable correction on stdlib-specific questions, leaving
> general-fortran questions unchanged.

If post-training Q3 produces a correct answer pointing at
`stdlib_sorting`, the LoRA added real domain knowledge above the base
capability — and dlm's product story stands on more than just style
adaptation.

If post-training Q3 still hallucinates and Q1/Q2/Q4 *degrade*, the
LoRA is parasitic at this base — same memorization-without-
generalization pattern as Findings 01 and 02, just at a more capable
base.

## Sample base outputs (verbatim, truncated)

```
Q1: How do you declare an allocatable array of real(real64) in modern Fortran?
> In modern Fortran, you can declare an allocatable array of real numbers
> using the `ALLOCATABLE` keyword. ...
> real(real64), allocatable :: my_array(:)
> allocate(my_array(5))
```

```
Q3: Show the signature of stdlib's sorting routine in Fortran.
> The standard library in Fortran does not have a built-in sorting
> routine like C or other languages. However, you can use the `sort`
> function from the `iso_fortran_env` module to sort arrays.
```

(Q3 is the wedge.)
