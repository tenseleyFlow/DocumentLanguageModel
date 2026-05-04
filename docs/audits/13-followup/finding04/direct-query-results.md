# Finding 04 — direct-query smoke results

`qwen2.5-coder-1.5b` + finding-04 LoRA (v0002, 400 steps, train loss
0.62, eval loss 0.047, 98.6% token accuracy on training data).

`--backend pytorch` is required: dlm's auto-routing picks `mlx` on
darwin-arm64 by default, but MLX can't load PEFT-format adapters and
silently runs the base. (See "Notes on bugs surfaced" below.)

## Seen Q/A pairs (in training set)

### Q3 wedge — "Show the signature of stdlib's sorting routine in Fortran."

**Base:** wrong (hallucinates `iso_fortran_env::sort`)
**Finding-03 LoRA:** worse (degenerate "should be able to sort..." loop)
**Finding-04 LoRA: ✓ correct (verbatim training answer):**

```fortran
use stdlib_sorting, only: sort
call sort(array[, reverse])
```
The `array` argument is `intent(inout)` and must be a rank-1 array of
an intrinsic numeric type... The optional `reverse` argument is a
scalar logical with `intent(in)`.

### intent(in)

✓ correct, matches training answer verbatim.

## Unseen Q/A — stdlib variants

### "How would I sort a real(dp) array using stdlib?"

**Partial generalization.** The model knows `use stdlib_sorting` (real
information from the training corpus) but invents a wrong call form
(`call stdlib_sorting::qsort`) and falls back to a manual sort loop.

```fortran
use stdlib_sorting
call stdlib_sorting::qsort     ! wrong syntax
integer :: idx(1:SIZE=my_array)
real(dp) :: my_array(SIZE)
real(dp) :: result(SIZE)
... [hand-rolled sort loop]
```

The model learned *that* `stdlib_sorting` exists; it didn't learn the
generic call form well enough to apply it to a question it hadn't
seen verbatim.

### "What is stdlib_strings::starts_with?"

**No generalization.** Falls back to Rust syntax (`use ... ::`, `if
pred { ... }`).

```
use stdlib_strings::starts_with;

if starts_with("hello world", "hel") {
    // ...
}
```

The training set didn't cover `stdlib_strings::starts_with`
specifically; the model's nearest pattern is its base-pretraining Rust
knowledge.

### "How do I read a CSV file in Fortran with stdlib?"

**Hallucinated stdlib API.** The model produces plausibly-shaped
Fortran-stdlib code with wrong details (invented `stdlib_io_read_table`,
`stdlib_types`, `using` instead of `use`):

```fortran
use stdlib_types
using stdlib_types::array      ! invalid
array(real, dim=(:)) :: data
call stdlib_io_read_table("path/to/file.csv", data)   ! function doesn't exist
```

The training had `loadtxt` for stdlib_io, but the question phrasing
("read a CSV") didn't trigger that pattern. Model invented a
plausible-looking API instead.

## Out-of-domain — does the LoRA preserve general capability?

### "What is the capital of France?"
✓ "Paris."

### "Write a Python list comprehension that filters even numbers."
✓
```python
even_numbers = [i for i in range(10) if i % 2 == 0]
print(even_numbers)  # prints [0, 2, 4, 6, 8]
```

General-capability preservation is excellent. cal_general regression
was 0% in Finding 03 already; Finding 04's INSTRUCTION-only training
maintains that.

## Net wedge score

| Question shape | Result | Verdict |
| --- | --- | --- |
| Seen exactly | verbatim correct | ✓ memorized cleanly |
| Unseen with overlap (sort a real(dp) array) | knows module, wrong API form | partial |
| Unseen, different module (stdlib_strings::starts_with) | falls back to Rust | none |
| Unseen, different module (CSV/loadtxt) | hallucinates plausible API | none |
| Out-of-domain (English/Python) | unchanged from base | ✓ preserved |

## Reading

The recipe-shape fix worked **for in-distribution questions** —
qwen-coder-1.5b + INSTRUCTION-only Q/A training produced an adapter
that reproduces trained answers correctly without breaking general
capability. The wedge that Finding 03 widened (Q3 stdlib sorting), this
recipe narrows: the answer is now correct.

What we did *not* get is generalization to nearby-domain questions
the model hadn't seen verbatim. With only 32 hand-authored Q/A pairs
across multiple modules, the model can memorize all of them (98.6%
token accuracy) but doesn't have enough surface area to abstract the
pattern. For "ask about stdlib_strings::starts_with" to land
correctly, that module's API form needs to be in the training data.

The tradeoff is dataset-size-dependent: more Q/A → broader
generalization. With ~32 pairs we get pinpoint Q/A reproduction; with
~300 pairs (extrapolating) we'd plausibly cover most stdlib modules
at functional density.

## Notes on bugs surfaced

1. **`src/dlm/replay/store.py:187`** — `parse_instruction_body` called
   without `_normalize_probe_markers`. Fixed in this branch (added
   the import + call). Without the fix, `--fresh` doesn't help: the
   replay store retains snapshots with raw `### Q !probe` headers and
   the parser rejects them on every retrain.
2. **MLX adapter loading silently fails for PEFT adapters.** dlm's
   default backend on darwin-arm64 is MLX. PEFT `adapter_model.safetensors`
   isn't a valid MLX adapter format; MLX appears to load the base and
   silently ignore the adapter. `--backend pytorch` works correctly.
   The user-visible failure mode is "trained model behaves like base"
   — easy to misread as "training didn't work."
