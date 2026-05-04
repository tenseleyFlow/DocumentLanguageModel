# Finding 03 — direct-query smoke results

Greedy decoding (`temp=0.0`, `max-tokens=150-200`).

## Q1 — allocatable real(real64) array

```
$ dlm prompt finding03/the-doc.dlm "How do you declare an allocatable array of real(real64)..."
To declare an allocatable array of real(real64) in modern Fortran, you can use
the `allocatable` keyword followed by the array name and its dimensions...
real(real64), allocatable :: my_array(:)
allocate(my_array(10))
do i = 1, size(my_array) ...
deallocate(my_array)
```

Verdict: **correct, ≈ same as base.**

## Q2 — do concurrent vs do

```
$ dlm prompt finding03/the-doc.dlm "When should you prefer \`do concurrent\`..."
In what scenarios might you encounter a situation where `do concurrent` is more
suitable than a plain `do` loop? Additionally, how can you modify a `do
concurrent` loop to ensure that it executes in a specific order, and what are
the potential consequences of not doing so?

```fortran
program concurrent_example
    implicit none
    integer :: i, j
    integer, allocatable :: array(:)
    ...
    ! Use a plain do loop
    do i = 1, 10
        do j =
```

Verdict: **worse than base.** Repeats the question rhetorically, dumps an
unrelated allocatable code example, doesn't actually answer.

## Q3 — stdlib sorting (THE WEDGE)

```
$ dlm prompt finding03/the-doc.dlm "Show the signature of stdlib's sorting routine in Fortran."
The signature should be similar to the one in C, but with Fortran-specific syntax.
The sorting routine should be able to sort an array of integers in ascending order.
The sorting routine should be able to sort an array of floating-point numbers in ascending order.
The sorting routine should be able to sort an array of characters in ascending order.
The sorting routine should be able to sort an array of strings in ascending order.
The sorting routine should be able to sort an array of arrays in ascending order.
... (degenerate enumeration loop, never mentions stdlib_sorting)
```

Verdict: **worse than base.** Base hallucinated `iso_fortran_env::sort`
(concrete but wrong). Trained adapter degenerates into a "should be able to
sort..." enumeration loop. The wedge gap was *not* closed; it widened.

## Q4 — intent(in)

```
$ dlm prompt finding03/the-doc.dlm "What does \`intent(in)\` mean..."
In Fortran, a dummy argument is a placeholder for an actual argument that is
passed to a subroutine or function. The `intent(in)` attribute specifies that
the argument should be passed by value and should not be modified within the
subroutine or function...
subroutine example_subroutine(input_value)
    implicit none
    integer, intent(in) :: input_value
```

Verdict: **correct, ≈ same as base.**

## Net wedge score

**0/4 questions improved.** 2/4 actively degraded (Q2, Q3).
The LoRA did not add stdlib knowledge; it added autoregressive
fortran-source-completion habits that interfere with Q/A behavior.
