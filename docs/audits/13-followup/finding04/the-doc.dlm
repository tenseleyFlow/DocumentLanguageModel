---
dlm_id: 01KQDWAHNJ7FD72EQ4J4FXBJ2V
dlm_version: 15
base_model: qwen2.5-coder-1.5b
training:
  adapter: lora
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  sequence_len: 1024
  micro_batch_size: 1
  grad_accum: 8
  learning_rate: 2.0e-4
  warmup_ratio: 0.1
  num_epochs: 100
---

# Finding 04 — Q/A-shape recipe (INSTRUCTION-only)

Hand-authored Q/A pairs over `fortran-lang/stdlib`, sourced from
`/tmp/stdlib_build/doc/specs/*.md`. No raw `sources:` directives —
every training row is a `messages` row routed through Qwen's chatml
template. The wedge to track is **Q3: signature of stdlib's sorting
routine.** Finding 03's adapter degenerated into "should be able to
sort..." enumeration; here the model should learn to point at
`stdlib_sorting :: SORT` with proper signature.

stdlib_sorting density is intentional (~60% of pairs): it's the wedge
target. Other modules (io, math, strings, array) provide variety so
the LoRA isn't a one-module specialist.

::instruction::

### Q
What module in fortran-lang/stdlib provides sorting routines?

### A
The `stdlib_sorting` module. It provides four overloaded subroutines:
`ORD_SORT`, `SORT`, `RADIX_SORT`, and `SORT_INDEX`, plus the parameters
`int_index` and `int_index_low` for indexing.
::instruction::

### Q
Show the signature of stdlib's sorting routine in Fortran.

### A
```fortran
use stdlib_sorting, only: sort
call sort(array[, reverse])
```
The `array` argument is `intent(inout)` and must be a rank-1 array of
an intrinsic numeric type (integer kinds, real kinds), `character(*)`,
`type(string_type)`, `type(bitset_64)`, or `type(bitset_large)`.
The optional `reverse` argument is a scalar logical with `intent(in)`.
::instruction::

### Q
What's the difference between `ord_sort` and `sort` in stdlib?

### A
`ord_sort` is intended for arrays that have significant runs of
decreasing or non-decreasing values — it can achieve `O(N)` performance
on already-sorted or reverse-sorted input. `sort` is intended for
arrays that are effectively unordered before the call.
`ord_sort` is stable; `sort` is not.
::instruction::

### Q
What is `radix_sort` in `stdlib_sorting` for?

### A
`radix_sort` sorts fixed-width intrinsic data types — integers and
reals. It exploits the bit representation of these types for non-
comparison-based sorting and can outperform `sort` on numeric data.
::instruction::

### Q
What does `sort_index` return?

### A
`sort_index` sorts the input array AND returns indices that map the
original (pre-sort) array to its sorted version. This lets you reorder
related arrays in the same way without re-sorting them — useful when
sorting one column of a tabular dataset and needing to keep the other
columns aligned.
::instruction::

### Q
What's the signature of `ord_sort`?

### A
```fortran
call ord_sort(array[, work, reverse])
```
- `array` : `intent(inout)`, rank-1 array of supported intrinsic type
- `work` (optional): `intent(out)`, rank-1 same type as `array`, at
  least `size(array)/2` elements — used as scratch memory to reduce
  stack pressure
- `reverse` (optional): scalar default-logical `intent(in)`; if
  `.true.`, sort in non-increasing order
::instruction::

### Q
What sorting algorithm does `ord_sort` use under the hood?

### A
A hybrid combining merge sort and insertion sort. For random input it
performs like merge sort with `O(N log N)` worst-case — typically
slower than `sort`. For input with long runs of monotonic values, it
falls back toward `O(N)`, which is its specific advantage over `sort`.
::instruction::

### Q
Is `stdlib_sorting`'s `sort` stable?

### A
No. The `sort` subroutine is not stable. If you need a stable sort, use
`ord_sort`, which is stable in the case of equal keys. `radix_sort` is
also not stable.
::instruction::

### Q
What types can `stdlib_sorting`'s `ord_sort` accept?

### A
Rank-1 arrays of any of: `integer(int8)`, `integer(int16)`,
`integer(int32)`, `integer(int64)`, `real(sp)`, `real(dp)`, `real(qp)`,
`character(*)`, `type(string_type)`, `type(bitset_64)`, or
`type(bitset_large)`.
::instruction::

### Q
How do you use `stdlib_sorting`'s `sort` to sort a real array in
descending order?

### A
```fortran
use stdlib_sorting, only: sort
real, allocatable :: arr(:)
arr = [3.14, 1.41, 2.72, 0.58]
call sort(arr, reverse=.true.)
! arr is now [3.14, 2.72, 1.41, 0.58]
```
::instruction::

### Q
What kind of integer does `int_index` evaluate to?

### A
`int_index` is set to `int64` from the `stdlib_kinds` module — a
64-bit integer kind for indexing large arrays. The companion
`int_index_low` is `int32`.
::instruction::

### Q
What's the difference between `sort_adjoint` and `sort_index`?

### A
Both are based on `ord_sort`. `sort_adjoint` re-orders a *second*
array of the same size using the same permutations applied to the
input. `sort_index` instead returns the index permutation itself,
which the caller can then apply to any number of related arrays.
::instruction::

### Q
What's the licensing situation for `stdlib_sorting`?

### A
The Fortran Standard Library overall is MIT-licensed, but
`ord_sort`, `sort_adjoint`, `sort_index`, and `sort` are translations of
Rust's `slice.rs` sort, which is Apache-2.0 / MIT dual-licensed.
`radix_sort` does not have this Rust ancestry.
::instruction::

### Q
Why might you pass the optional `work` argument to `ord_sort`?

### A
`work` is used as scratch memory for `ord_sort`'s internal record
keeping. Passing a pre-allocated `work` array (e.g. one in static
storage) avoids stack allocation during the sort, which can
significantly reduce stack memory pressure when sorting large arrays
inside deeply nested call chains.
::instruction::

### Q
What happens if you `ord_sort` a real array containing `NaN`?

### A
The order of the result is undefined. The same applies to `sort`,
`sort_adjoint`, `sort_index`, and `radix_sort`. If your data may
contain NaN values, filter them out before calling stdlib's sort
routines.
::instruction::

### Q
Show how to sort an integer array AND get indices to reorder a
companion array using `stdlib_sorting`.

### A
```fortran
use stdlib_sorting, only: sort_index, int_index
integer :: keys(5) = [3, 1, 4, 1, 5]
real    :: vals(5) = [10.0, 20.0, 30.0, 40.0, 50.0]
integer(int_index) :: idx(5)
call sort_index(keys, idx)        ! keys is now sorted; idx maps original->sorted
vals = vals(idx)                  ! reorder vals using the same permutation
```
::instruction::

### Q
What are the four overloaded subroutines in `stdlib_sorting`?

### A
`ORD_SORT`, `SORT`, `RADIX_SORT`, and `SORT_INDEX`. (`SORT_ADJOINT`
exists but the spec groups it under `ORD_SORT`'s family.)
::instruction::

### Q
Are stdlib's sort routines thread-safe?

### A
Each call modifies its `intent(inout)` `array` argument in place, so
two threads must not call sort on the same array concurrently. With
distinct arrays, there's no shared state in the sort routines
themselves — they're reentrant.
::instruction::

### Q
What is the `stdlib_io` module for in fortran-lang/stdlib?

### A
`stdlib_io` provides text-file IO routines — primarily `loadtxt` for
reading delimited numeric data into rank-2 arrays and `savetxt` for
writing them. It also exposes formatting helpers like `disp` for
human-readable display of arrays and scalars.
::instruction::

### Q
Show the signature of `stdlib_io::loadtxt`.

### A
```fortran
use stdlib_io, only: loadtxt
call loadtxt(filename, array[, skiprows, max_rows, fmt])
```
The `array` is allocated and filled from the file; supported types
include `real(sp)`, `real(dp)`, and `complex(dp)`. `skiprows` skips
header lines, `max_rows` caps the read.
::instruction::

### Q
What's `stdlib_string_type`'s purpose?

### A
It defines `type(string_type)` — a deferred-length, allocatable string
wrapper around `character(:)` — and overloads operators (`==`, `<`,
`//`) for it. This gives Fortran a value-semantic variable-length
string type that's ergonomic in containers and arrays, working around
the limitations of bare `character(*)`.
::instruction::

### Q
What does `stdlib_array` provide?

### A
`stdlib_array` provides index-manipulation utilities — `trueloc` and
`falseloc` for finding indices where a logical array is true or false,
plus indexing helpers. Useful for masking and selecting elements
without writing manual loops.
::instruction::

### Q
What is `stdlib_math::linspace`?

### A
`linspace(start, end, n)` returns a rank-1 array of `n` evenly-spaced
values from `start` to `end` inclusive. Same shape as numpy's `linspace`.
The companion `logspace(start, end, n[, base])` returns evenly-spaced
values in log space.
::instruction::

### Q
What is `stdlib_stats::mean` for?

### A
Computes the arithmetic mean of a rank-N array along an optional
specified dimension. With `dim` omitted, returns a scalar mean over
all elements. With `dim=k`, returns a rank-(N-1) array of means along
the k-th axis. Optionally accepts a `mask` for conditional inclusion.
::instruction::

### Q
What's `stdlib_linalg::solve` for?

### A
Solves a dense linear system `A . x = b` for `x`, where `A` is a
square matrix and `b` is a right-hand-side vector or matrix. Wraps
LAPACK's `gesv` family. Returns the solution; the input `A` is
typically modified in-place by the LU decomposition.
::instruction::

### Q
What `kind` parameters does `stdlib_kinds` define?

### A
`int8`, `int16`, `int32`, `int64` for integer kinds and `sp`, `dp`,
`qp` for real kinds (single, double, quadruple precision). Also
`xdp` for extended-double where supported. These wrap the standard
`iso_fortran_env` kinds with shorter names.
::instruction::

### Q
What does `stdlib_ascii::to_upper` do?

### A
Returns its `character(*)` argument with ASCII letters (a-z) mapped to
their uppercase equivalents (A-Z). Non-letter characters pass through
unchanged. The companion `to_lower` does the opposite.
::instruction::

### Q
What's `stdlib_strings::replace_all`?

### A
```fortran
use stdlib_strings, only: replace_all
result = replace_all(string, pattern, replacement)
```
Returns `string` with every occurrence of `pattern` substituted by
`replacement`. Pure and elemental, so it can be applied to scalar or
array string inputs.
::instruction::

### Q
What's the difference between `stdlib_random::random_seed` and
Fortran's intrinsic `random_seed`?

### A
The intrinsic `random_seed` configures the implementation-dependent
default RNG. `stdlib_random::random_seed` is part of stdlib's
deterministic RNG path — it seeds stdlib's portable distribution
sampling routines, giving reproducible random sequences across
compilers.
::instruction::

### Q
What does `stdlib_hashmaps::hashmap_type` provide?

### A
A generic open-addressing hash map keyed by integer or string and
storing arbitrary user-defined data. `set`, `get`, `remove`, `keys`,
and iterators are provided as type-bound procedures.
::instruction::

### Q
What's `stdlib_quadrature::trapz` for?

### A
Numerical integration via the trapezoidal rule. `trapz(y, x)` returns
the integral of y dx using sample points `x` and corresponding values
`y`. Both must be rank-1 arrays of the same length. `simps` is the
cousin using Simpson's rule.
::instruction::

### Q
How does the `stdlib_logger` module emit log messages?

### A
Define a `logger_type` instance, configure its level (`debug`, `info`,
`warning`, `error`), then call methods like `log_information`,
`log_warning`, etc. Output goes to the configured unit (default
stderr). Logger instances can be configured with timestamps, source
locations, and log-level filtering.
::instruction::

### Q
What does `intent(in)` mean for a Fortran dummy argument?

### A
The argument may be read but not modified inside the procedure.
Attempting to assign to or pass it as `intent(out)`/`intent(inout)`
to another routine is a compile-time error (in conformant compilers).
This is the contract you want for read-only inputs.
::instruction::

### Q
When should you prefer `do concurrent` over a plain `do` loop?

### A
When loop iterations have no carried dependence — that is, no
iteration reads or writes a variable that another iteration could
write. `do concurrent` expresses this independence to the compiler,
which can then auto-vectorize, parallelize via OpenMP, or offload to
GPU without further pragmas. If iterations are actually dependent,
use plain `do`.
::instruction::

### Q
What's the canonical way to declare an allocatable array of
`real(real64)` in modern Fortran?

### A
```fortran
use iso_fortran_env, only: real64
real(real64), allocatable :: arr(:)
allocate(arr(n))
! ... use arr ...
deallocate(arr)
```
Or use stdlib's `dp` kind:
```fortran
use stdlib_kinds, only: dp
real(dp), allocatable :: arr(:)
```
