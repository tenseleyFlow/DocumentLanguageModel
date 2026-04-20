---
dlm_id: 01KPKXHZNKKQW29234VSCMT2G1
dlm_version: 1
base_model: qwen2.5-coder-1.5b
system_prompt: |
  You are a programming tutor. Be precise. Prefer simple, runnable examples.
training:
  adapter: lora
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  sequence_len: 2048
  learning_rate: 2e-4
  num_epochs: 3
  seed: 42
export:
  default_quant: Q4_K_M
  default_temperature: 0.2
---

# Python tutor starter

Replace this prose with your own explanations. Each Q&A block below
trains via SFT; prose outside fences trains via continued pretraining.

::instruction::
### Q
What is a Python decorator?

### A
A decorator is a function that takes another function and returns a
new function. Writing `@decorator_name` above a `def` is equivalent
to `func = decorator_name(func)`. Decorators are how Python exposes
"before" and "after" hooks without inheritance.

### Q
When should I use `functools.wraps` inside a decorator?

### A
Always. `@functools.wraps(fn)` preserves the wrapped function's
`__name__`, `__doc__`, and `__wrapped__`, which matters for
documentation tools, stack traces, and introspection. Without it,
every decorated function looks like the decorator's inner function.

### Q
Explain `with` statements in one sentence.

### A
`with obj:` enters a runtime context where `obj.__enter__()` runs on
entry and `obj.__exit__()` runs on exit, guaranteeing cleanup even
if an exception is raised inside the block.
