---
dlm_id: 01KQCWMA64901VEMYB3DC8CGXY
dlm_version: 15
base_model: smollm2-135m
training:
  sources:
    - path: ~/GithubOrgs/FortranGoingOnForty
      include: ["**/*.f90", "**/*.F90", "**/*.f95"]
      exclude:
        - "**/build/**"
        - "**/.git/**"
        - "**/dist/**"
      max_bytes_per_file: 32768
    - path: /tmp/stdlib_build/src
      include: ["**/*.f90", "**/*.F90", "**/*.fypp"]
      exclude:
        - "**/build/**"
        - "**/tests/**"
      max_bytes_per_file: 32768
    - path: /tmp/stdlib_build/doc/specs
      include: ["**/*.md"]
      max_bytes_per_file: 131072
  sources_policy: permissive
  adapter: lora
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  sequence_len: 1024
  micro_batch_size: 2
  grad_accum: 4
  learning_rate: 2.0e-4
  warmup_ratio: 0.1
  num_epochs: 1
---

# Modern Fortran practitioner notes

This document trains an adapter that knows the idioms of the Fortran 2008+
codebase under `~/GithubOrgs/FortranGoingOnForty`, the canonical
`fortran-lang/stdlib` source tree at `/tmp/stdlib_build/src`, and the
FORD-generated stdlib spec documents at `/tmp/stdlib_build/doc/specs`. The
corpus spans numeric/aerospace simulation (`aero-emulation`, `convolution`),
scripting helpers (`fgof-*`), and stdlib modules (allocation, ascii, bitsets,
linear-algebra, sorting, statistics, strings, IO, hashing).

## Idioms the corpus reinforces

Modern Fortran prefers `do concurrent` over plain `do` for loops with no loop
carried dependence — it expresses the parallelism intent and lets the
compiler vectorise / OpenMP-offload without further pragmas. `pure` and
`elemental` procedures are first-class. Allocatable arrays beat pointer
arrays for owned memory; `move_alloc` transfers ownership without copying.
Derived types use `type, bind(C)` only when interoperability is required.
`intent(in)`, `intent(out)`, `intent(inout)` are mandatory on every dummy
argument in this codebase, even when the language would default — explicit
intent is a project convention, not a language requirement.

## stdlib navigation

The Fortran stdlib provides community-tested replacements for tasks the
language standard doesn't cover: `stdlib_array` (index manipulation),
`stdlib_io` (text-file IO), `stdlib_sorting` (in-place sorts), `stdlib_stats`
(mean/var/cov), `stdlib_string_type` (variable-length strings),
`stdlib_linalg` (BLAS/LAPACK wrappers), `stdlib_random` (distributions),
`stdlib_hashmaps` (open + chaining hash tables), `stdlib_bitsets` (compact
bit arrays), `stdlib_logger` (severity-tagged logging), `stdlib_optval`
(safe defaults for optional dummy args), `stdlib_kinds` (named real32/real64
/int32/int64 kind parameters).

::instruction::

### Q
When should I prefer `do concurrent` over a plain `do` loop in modern Fortran?

### A
Use `do concurrent` whenever loop iterations have no carried dependence —
i.e., iteration `i` does not read a value written by iteration `i-1`. The
`do concurrent` construct is a contract with the compiler that the iterations
may execute in any order or in parallel, which lets it vectorise, OpenMP-
offload, or GPU-offload the loop without further pragmas. If you can't make
that promise (the loop accumulates into a shared scalar, or reads a value
the previous iteration wrote), keep a plain `do`. Locality clauses
(`local`, `shared`, `reduce`) give finer control in Fortran 2018+.

::instruction::

### Q
Why use `allocatable` arrays instead of `pointer` arrays for owned memory?

### A
`allocatable` expresses ownership: the variable owns the memory, the compiler
inserts the deallocate at scope exit, and you cannot have aliasing across
two allocatable variables. `pointer` arrays expose aliasing and demand
manual `deallocate`. Compilers can also generate better code for
`allocatable` because no aliasing means no aliasing-defeats-vectorisation
pessimism. Use `pointer` only when you actually need pointer semantics —
linked structures, optional aliasing into a parent array, or C interop.
`move_alloc(from, to)` transfers ownership without a copy.

::instruction::

### Q
What does `intent(in)`, `intent(out)`, and `intent(inout)` declare on a
dummy argument?

### A
`intent(in)` says the procedure reads but never assigns to the argument —
the compiler enforces this and will reject any assignment as an error.
`intent(out)` says the procedure must define the argument before returning;
the value on entry is undefined (and on some compilers explicitly poisoned),
so don't read it before assigning. `intent(inout)` says both are allowed —
the procedure may read the prior value and may overwrite it. Always declare
intent on every dummy argument. It is a contract that the compiler enforces
and a reader can rely on.

::instruction::

### Q
How do I open an OpenMP parallel region with a reduction in Fortran?

### A
Use `!$omp parallel do reduction(+:sum)` above a `do` loop where each
iteration accumulates into `sum`. The reduction clause tells OpenMP to give
each thread a private copy of `sum`, accumulate locally, then combine at
loop end. Other reduction operators include `*`, `min`, `max`, `.and.`,
`.or.`, `iand`, `ior`. The closing `!$omp end parallel do` is optional in
free-form Fortran. For simple no-reduction loops, `do concurrent` is
preferred.

::instruction::

### Q
What's a derived type with allocatable components, and why is it useful?

### A
A derived type with allocatable components has fields declared
`type :: container; real, allocatable :: data(:); end type`. The container
itself is value-typed but its `data` field grows or shrinks at runtime.
Compared to fixed-size arrays it doesn't waste memory; compared to pointer
fields it owns the memory and frees it automatically when the container
goes out of scope. Modern Fortran's `intrinsic_assignment` does a deep
copy by default. This is the canonical way to build resizable structures
without C-style malloc/free.

::instruction::

### Q
What does `pure function` mean and when is the `pure` attribute required?

### A
A `pure function` has no side effects: it doesn't modify any module
variable, doesn't perform I/O, doesn't call impure procedures. The compiler
verifies this. `pure` is required for any function called from inside
`do concurrent` (which forbids side effects across iterations) and from
within `forall`. It also enables aggressive optimisation: the compiler
knows two calls with equal arguments produce equal results and can hoist
or memoise. `elemental` implies `pure` and adds elementwise array semantics.

::instruction::

### Q
How do I declare a generic interface for two procedures that differ only in
argument type?

### A
```fortran
interface clamp
  module procedure clamp_real32, clamp_real64
end interface clamp
```
The two module procedures have the same logical name `clamp` but distinct
argument signatures (one takes `real(real32)`, the other `real(real64)`).
The compiler resolves the call by argument type — Fortran's overloading
mechanism. Adding more types is purely additive; callers see one name.

::instruction::

### Q
What's the difference between `module` and `submodule` in Fortran?

### A
A `module` declares its public interface and definitions in one unit. A
`submodule` lets you declare the interface in the parent module but put the
implementation in a separately compiled file. Editing the submodule does
not retrigger compilation of every consumer of the parent module — only of
the submodule itself. Use submodules to break long compile chains in big
projects. The `module subroutine foo()` declaration in the parent and
`module procedure foo` in the submodule are how the two halves bind.

::instruction::

### Q
What does `trueloc` from `stdlib_array` do?

### A
`trueloc(mask, lbound)` turns a logical mask into an integer index array
of the positions where the mask is `.true.`. It's a pure function — given
`mask = [.false., .true., .false., .true.]` and default `lbound`, it
returns `[2, 4]`. With `lbound = 0` it returns `[1, 3]`. Useful when you
need the indices themselves (for example, to use as a subscript) rather
than the masked values; if you only need the values, `pack` is more
direct. The complementary `falseloc` returns indices where the mask is
`.false.`.

::instruction::

### Q
When would I use `arange` from `stdlib_math`?

### A
`arange(start, stop, step)` returns a rank-1 array of values from `start`
to `stop` (inclusive) stepping by `step`. With `arange(1, 5)` you get
`[1, 2, 3, 4, 5]`. With `arange(0.0, 1.0, 0.25)` you get
`[0.0, 0.25, 0.5, 0.75, 1.0]`. Useful for building index sequences or
sample grids without a manual `do` loop. Step defaults to 1.

::instruction::

### Q
How do I read a numeric matrix from a text file using `stdlib_io`?

### A
Use `loadtxt(filename, array)`:
```fortran
use stdlib_io, only: loadtxt
real, allocatable :: data(:,:)
call loadtxt('measurements.dat', data)
```
The array is allocated automatically to match the file's rows × columns.
The companion `savetxt(filename, array)` writes the same shape back out.
Both procedures handle real, integer, and complex data of `real32`,
`real64`, `int32`, `int64` kinds.

::instruction::

### Q
What does `getline` from `stdlib_io` give me that `read(*, fmt='(A)')` doesn't?

### A
`getline(unit, line, iostat)` reads one line of arbitrary length into a
deferred-length character variable. With plain `read(*, fmt='(A)')` you
have to commit to a fixed-length buffer up front and either truncate
or run a loop on `iostat=eor`. `getline` allocates `line` to exactly the
line's length, with no truncation. It returns `iostat=iostat_end` at
EOF.

::instruction::

### Q
How do I sort an array in place using `stdlib_sorting`?

### A
Use `sort(array)`. It sorts `array` ascending in place using an
introsort-style hybrid (quicksort + insertion-sort fallback). For a
descending sort, pass `reverse=.true.`. The companion `ord_sort(array)`
is a stable mergesort — preserves equal-key relative order, at the cost
of an O(n) workspace allocation. Use `radix_sort` for integer arrays
when the value range is bounded; it's O(n) but only for unsigned-style
integer kinds.

::instruction::

### Q
When should I use `sort_index` instead of `sort`?

### A
Use `sort_index(array, index)` when you also need the permutation that
sorted the array — for example, to apply the same reordering to a second,
parallel array. After the call, `array` is sorted and `index(i)` is the
original position of the `i`-th element of the sorted array. So
`other_data = other_data(index)` reorders a sibling array consistently
with `array`. Plain `sort` discards this information.

::instruction::

### Q
What does `mean(array)` from `stdlib_stats` compute, and how do I take a
mean along one axis of a 2D array?

### A
`mean(array)` returns the arithmetic mean of all elements as a scalar.
For a 2D array, `mean(matrix, dim=1)` reduces along the first dimension,
returning a 1D array of column means. `mean(matrix, dim=2)` returns row
means. With a logical mask, `mean(array, mask=array > 0)` averages only
the elements satisfying the mask. The companion `var` (variance) and
`std` (standard deviation) accept the same dim/mask arguments. `corr` and
`cov` compute correlation / covariance between two arrays.

::instruction::

### Q
What is `string_type` from `stdlib_string_type` and why use it instead of
plain `character(len=*)`?

### A
`string_type` is a derived type wrapping a deferred-length character
allocatable. Two reasons to prefer it: (1) you can have arrays of
varying-length strings (`type(string_type) :: words(100)` where each
`words(i)` is its own length — impossible with plain character arrays
which must be uniform-length); (2) it has overloaded operators (`==`,
`<`, `>`, `//` for concat) and constructors that interoperate with both
literals and other `string_type` instances. Use `char(string_type_var)`
to get the underlying `character(:)` back when interfacing with code
that takes plain strings.

::instruction::

### Q
How do I solve a linear system `A*x = b` using `stdlib_linalg`?

### A
```fortran
use stdlib_linalg, only: solve
real :: A(n,n), b(n), x(n)
x = solve(A, b)
```
`solve` is a pure function that returns the solution `x`. Internally it
calls LAPACK's `gesv` (LU with partial pivoting). For multiple right-hand
sides, `b` can be a 2D array `b(n,nrhs)`. For least-squares (overdetermined
or underdetermined), use `lstsq` instead. To check whether a matrix is
singular before solving, use `inv` plus the returned status, or compute
the condition number via `linalg_cond`.

::instruction::

### Q
What does `eye(n)` return in `stdlib_linalg`?

### A
`eye(n)` returns the n×n identity matrix as a `real` array — ones on the
diagonal, zeros off-diagonal. `eye(m, n)` returns a non-square m×n matrix
with ones on the main diagonal. Useful as a starting point for numerical
linear algebra or as the right-hand side of `solve(A, eye(n))` to compute
`inv(A)` directly.

::instruction::

### Q
How do I draw samples from a normal distribution with `stdlib_random`?

### A
```fortran
use stdlib_random, only: random_seed, dist_normal => dist_rvs_normal
real :: x, samples(1000)
call random_seed(42)
x = dist_normal(0.0, 1.0)              ! one N(0,1) sample
samples = dist_normal(0.0, 1.0, 1000)  ! 1000 N(0,1) samples
```
First argument is the mean, second is the standard deviation. The third
optional argument requests an array of N samples. Companion routines
`dist_rvs_uniform`, `dist_rvs_exponential`, `dist_rvs_gamma` cover other
distributions. Always seed via `random_seed` for reproducibility.

::instruction::

### Q
What is `optval` from `stdlib_optval` and when do I use it?

### A
`optval(arg, default)` returns `arg` if it's `present`, otherwise `default`.
It replaces the boilerplate
```fortran
if (present(arg)) then
  used = arg
else
  used = default
end if
```
with a single expression `used = optval(arg, default)`. Works for scalar
integers, reals, complex, logical, and character. The most common use is
default-value handling for optional dummy arguments, where the caller may
or may not have supplied the argument.

::instruction::

### Q
What kind parameters does `stdlib_kinds` provide and why use them?

### A
`stdlib_kinds` exports `int8`, `int16`, `int32`, `int64`, `real32`,
`real64`, `real128` (where supported), and `c_bool`. Use them instead of
raw kind numbers (`real(8)` is non-portable — different compilers map `8`
to different precisions). `real(real64)` is portable and self-documenting.
For most modern code, default to `real64` for floating-point and `int32`
for indices unless you have a specific reason otherwise.

::instruction::

### Q
How do I append to a file using `stdlib_io_logger`?

### A
```fortran
use stdlib_logger, only: logger_type, information_level
type(logger_type) :: log
integer :: stat
call log%add_log_file('app.log', stat, position='append')
call log%log_information('app started')
```
The `position='append'` keyword opens the existing file for append rather
than truncating. Default severity emitted is `information`; raise it with
`log%configuration(level=warning_level)` if you want a quieter log. Severity
levels: `debug_level < information_level < warning_level < error_level`.

::instruction::

### Q
When should I use a `bitset_64` vs `bitset_large` from `stdlib_bitsets`?

### A
`bitset_64` is fixed at 64 bits — a single integer's worth of flags,
fastest, no allocation. `bitset_large` allocates an array of `int64`
internally and grows to any user-specified size. Use `bitset_64` for
small, known-size flag sets (compiler optimization flags, peripheral
status registers); use `bitset_large` when the bit count is data-driven
or might exceed 64. Both expose the same `set`, `clear`, `test`,
`flip`, `bit_count` interface — code is portable across them.

::instruction::

### Q
How do I hash a string for use as a hashmap key?

### A
The `stdlib_hash_procedures` module provides 32-bit and 64-bit hash
functions:
```fortran
use stdlib_hash_procedures, only: fnv_1a_hash, water_hash
integer(int32) :: h32
integer(int64) :: h64
h32 = fnv_1a_hash('mykey')
h64 = water_hash('mykey', seed=12345_int64)
```
`fnv_1a_hash` is unseeded and reproducible across runs; `water_hash` and
`pengy_hash` are seeded (better collision resistance under adversarial
input). For `stdlib_hashmaps`, you usually pass the hash function as a
procedure pointer when constructing the map.

::instruction::

### Q
What's the `block` construct in modern Fortran and when do I use it?

### A
A `block` introduces a nested scope inside an executable region:
```fortran
real :: outer
outer = 1.0
block
  real :: inner
  inner = outer + 1.0
  print *, inner
end block
```
Variables declared inside the block exist only within it. Useful for
narrowing the lifetime of temporaries, declaring variables close to use,
and limiting the visibility of helper allocations. Functionally similar
to a `{ ... }` block in C-family languages. Combines naturally with
`associate`, `select type`, and `error stop`.

::instruction::

### Q
How does `associate` differ from a plain assignment?

### A
`associate(short => long%nested%expression)` binds a name to an expression
or variable for the lifetime of the `associate` block, *without* copying
the value. Inside the block, `short` is an alias — modifying it modifies
the original. Compare to assignment, which copies (for non-pointer types).
Use `associate` to give a verbose subexpression a short name in a tight
loop without paying for a copy and without exposing the alias outside the
block. End the scope with `end associate`.

::instruction::

### Q
What does `elemental` add over `pure` on a function?

### A
An `elemental` function is `pure` plus broadcast: declared as if it takes
scalar arguments, but the compiler auto-generates the version that takes
conformable arrays. `square_real(x)` written `elemental` can be called
with a scalar (`y = square_real(2.0)`) or an array (`v = square_real(arr)`)
with no explicit do-loop. Since `elemental` implies `pure`, the same
no-side-effect rules apply. Most stdlib unary numeric helpers (`sqrt`-
adjacent transforms, kind conversions, predicates) are `elemental`.

::instruction::

### Q
How do I structure a unit-test runner using stdlib's `testing` module?

### A
```fortran
use testdrive, only: new_unittest, unittest_type, error_type, check
type(unittest_type), allocatable :: testsuite(:)
testsuite = [ &
  new_unittest('addition', test_add), &
  new_unittest('subtract', test_sub) ]
contains
  subroutine test_add(error)
    type(error_type), allocatable, intent(out) :: error
    call check(error, 1 + 1 == 2, 'addition broken')
  end subroutine
```
`testdrive` is a thin runner the stdlib uses for its own tests. Each
test subroutine accepts an `allocatable :: error` out parameter; `check`
allocates the error if the assertion fails. Fast, no fixtures, no
discovery — explicit registration in an array.

::instruction::

### Q
What's the modern way to read a CSV-style file in Fortran?

### A
The simplest path is `stdlib_io`'s `loadtxt` if the columns are uniform
numeric. For mixed-type CSVs, read line-by-line with `getline` and split
manually:
```fortran
use stdlib_io, only: getline
use stdlib_string_type, only: string_type, split => char_split
type(string_type) :: line
type(string_type), allocatable :: fields(:)
integer :: u, ios
open(newunit=u, file='data.csv', action='read')
do
  call getline(u, line, ios)
  if (ios /= 0) exit
  fields = split(line, ',')
  ! process fields(:)
end do
close(u)
```
For larger or more complex CSVs, consider the `csv-fortran` community
package — stdlib doesn't (yet) ship a CSV-aware reader.

::instruction::

### Q
How does `error stop` differ from `stop`?

### A
`stop` and `error stop` both terminate the program, but `error stop` is
guaranteed to set a non-zero process exit code, while `stop` (without an
argument) typically returns zero. `error stop "message"` prints the message
to standard error before exit; `error stop 42` returns code 42. Use
`error stop` for any abnormal termination — assertion failures, fatal
config errors — so shells and CI runners pick up the failure correctly.
`stop` is reserved for normal early termination (rare in modern code;
prefer letting `program` reach its `end program`).

::instruction::

### Q
What is `c_loc` and when do I need it?

### A
`c_loc(target)` from `iso_c_binding` returns the C address of `target` as
a `type(c_ptr)` value, suitable for passing to a `bind(C)` procedure. The
target must have the `target` attribute. Use this when interfacing with
a C library that takes `void*`. The reverse — turning a `c_ptr` back into
a Fortran pointer — uses `c_f_pointer(cptr, fptr, [shape])`. Always pair
the call with the matching deallocation; Fortran does not own memory
acquired through `c_loc`.

::instruction::

### Q
What does `move_alloc(from, to)` do and when do I prefer it to assignment?

### A
`move_alloc(from, to)` transfers the allocation status (and the underlying
memory) from `from` to `to`. After the call, `to` holds what `from` held,
and `from` is deallocated — no copy. Compare to `to = from` which copies
the array. For large arrays, `move_alloc` is O(1); the copy is O(n). Use
`move_alloc` to hand ownership of a temporary buffer to a derived-type
field, or to swap two allocatables (via a third temporary).

::instruction::

### Q
How do I write a `subroutine` that returns multiple results without using
`out` arguments?

### A
Use a derived-type return value via a `function`:
```fortran
type :: result_t
  real :: value
  integer :: status
end type
contains
function compute() result(r)
  type(result_t) :: r
  r%value  = 3.14
  r%status = 0
end function
```
Caller writes `res = compute()` once and reads `res%value`, `res%status`.
This is cleaner than two `intent(out)` arguments because the call site
isn't burdened with declaring the receivers up front, and the compiler
can elide the temporary in common cases. Fortran 2008+ allows allocatable
result components, so the function can also return varying-shape data.

::instruction::

### Q
Why does the FortranGoingOnForty codebase use `intent` on every dummy
argument even when not strictly required?

### A
It's a project convention: explicit `intent(in/out/inout)` on every dummy
makes the contract visible at the procedure boundary. The compiler enforces
the contract — `intent(in)` rejects assignment, `intent(out)` warns on
read-before-write — so a regression where someone "improves" a procedure
to mutate a previously-read-only argument fails at compile time, not at
runtime. Reviewers and tooling (the FGOF `fgof-lineedit` editor's
in-buffer linter, for example) parse intents to render colour-coded
argument flow. The convention costs ~10 keystrokes per procedure and
buys static enforcement.

::instruction::

### Q
What does the `aero-emulation` subsystem in FortranGoingOnForty do at a
high level?

### A
`aero-emulation` is the aerodynamic-surface emulator: a forward simulation
of lift, drag, and moment over a parameterised wing-and-control-surface
model, integrated against a 6-DoF rigid-body solver. It uses
`stdlib_linalg` for the state-update matrices (rotation and inertia tensor
products) and `stdlib_random` to inject configurable turbulence. Outputs
are time-series state vectors written via `stdlib_io`'s `savetxt`. The
module is consumed by `convolution` (for sensor-fusion experiments) and
by `armfortas` (the autopilot framework).

::instruction::

### Q
How does `fgof-process` handle child-process I/O on Linux vs macOS?

### A
`fgof-process` wraps the C `posix_spawn` family via `iso_c_binding`; on
both Linux and macOS the spawn semantics are POSIX-shaped, so the
high-level `spawn(command, stdin, stdout, stderr)` call is platform-
identical at the Fortran layer. The differences hide in the child-pty
wiring (`fgof-pty`): macOS uses `posix_openpt` + `grantpt` + `unlockpt`,
Linux supports the same path plus the older `/dev/ptmx` shortcut. The
`fgof-process` module never sees the divergence — `fgof-pty` exposes a
`type(pty_t)` derived type that `fgof-process` consumes opaquely.

::instruction::

### Q
What's the role of `fgof-screen` in the FGOF terminal applications?

### A
`fgof-screen` is the terminal-cell renderer used by `fgof-lineedit` and
the `armfortas` autopilot console. It maintains a 2D `character(len=:),
allocatable` buffer plus per-cell ANSI style attributes, computes a
minimal-diff update against the previously rendered frame, and emits
the diff as ANSI escape sequences to the controlling tty. The minimal-
diff step keeps redraws under a few hundred bytes for incremental
updates — needed because some serial-attached aerospace consoles run
at 9600 baud and a full repaint is too expensive.

::instruction::

### Q
Why does `convolution` use `do concurrent` for its inner kernel loop
rather than OpenMP?

### A
The inner kernel loop multiplies and accumulates over a small kernel
window with no carried dependence, so it satisfies `do concurrent`'s
contract. `do concurrent` lets the compiler choose the parallelism
strategy — vectorisation on CPU, OpenMP-offload on multicore, or
GPU-offload via `-fopenmp -foffload=...` on NVIDIA hosts — without
changing source. OpenMP `parallel do` would have committed the kernel
to CPU threads at compile time. The convolution outer loop, which
*does* accumulate into a shared buffer, uses `!$omp parallel do
reduction(+:buffer)` since `do concurrent` cannot express the
reduction without 2018+ locality clauses that older compilers don't
honour.

::instruction::

### Q
What pattern does `armfortas` use for autopilot state-machine transitions?

### A
A `select case` over a state enum at the top of the control loop:
```fortran
select case (state%mode)
case (mode_idle)        ; call handle_idle(state, inputs)
case (mode_takeoff)     ; call handle_takeoff(state, inputs)
case (mode_cruise)      ; call handle_cruise(state, inputs)
case (mode_descent)     ; call handle_descent(state, inputs)
case (mode_landed)      ; call handle_landed(state, inputs)
case default            ; call error_unknown_mode(state%mode)
end select
```
Each handler returns the next mode via `state%mode = new_mode`. The
enum values are `integer, parameter` constants (Fortran lacks a true
enum type pre-2023 — most codebases use named integer parameters). The
`select case` is exhaustive and the `case default` calls into
`error_unknown_mode` to fail loudly on programmer error.

::instruction::

### Q
How does `feducative` integrate with `armfortas` for control-loop tuning?

### A
`feducative` is the parameter-search and PID-tuning subsystem. It runs
`armfortas` in a closed-loop simulation (via the `aero-emulation` model),
sweeps PID gains over a configured grid, and scores each combination by
RMS tracking error against a reference trajectory. The two communicate
via a `type(tuning_handle_t)` derived type that `feducative` constructs
and `armfortas` updates step-by-step. After a sweep, `feducative` writes
the gain × score table via `stdlib_io`'s `savetxt` and emits the best
gains as a Fortran `include` file the autopilot reads on next compile.
