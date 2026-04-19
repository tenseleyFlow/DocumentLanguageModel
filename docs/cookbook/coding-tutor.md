# Coding tutor

Build a programming Q&A model that answers questions in your team's
voice. Roughly 5 minutes of edit-train-prompt iteration.

## Goal

A `.dlm` trained on instruction Q&A pairs that explains Python
concepts using your house style and idioms.

## Starter template

Copy `templates/coding-tutor.dlm` from the repo, or start from
scratch:

```dlm
---
dlm_id: 01HRCODING0000000000000000     # dlm init replaces this
base_model: qwen2.5-coder-1.5b
system_prompt: |
  You are a Python tutor. Be precise. Prefer simple examples.
training:
  lora_r: 16
  learning_rate: 2e-4
  num_epochs: 3
export:
  default_quant: Q4_K_M
  default_temperature: 0.2
---

# Python coding tutor

::instruction::
### Q
What is a decorator in Python?

### A
A decorator is a function that takes a function and returns a new
function. The `@decorator_name` syntax above `def foo(): ...` is
equivalent to writing `foo = decorator_name(foo)`.

### Q
When should I use `functools.wraps` inside a decorator?

### A
Always. Without it, the wrapped function loses its `__name__`,
`__doc__`, and `__wrapped__` — introspection and debugging get
confused, and Sphinx / mkdocstrings can't find the real docstring.
```

## Walk-through

```sh
# Create the document
$ uv run dlm init tutor.dlm --base qwen2.5-coder-1.5b

# Paste the Q&A above into tutor.dlm

# Train
$ uv run dlm train tutor.dlm --max-steps 50
trained: v0001 (50 steps, seed=42, determinism=strong)

# Smoke-test via HF inference
$ uv run dlm prompt tutor.dlm "Explain closures in one sentence."
A closure is an inner function that captures variables from its
enclosing scope so those variables stay alive after the outer call
returns.

# Ship to Ollama
$ uv run dlm export tutor.dlm --name coding-tutor
ollama: registered coding-tutor:latest

# Use it
$ ollama run coding-tutor "When should I use list vs tuple?"
Lists are mutable — use them when the sequence changes. Tuples are
immutable — use them for fixed records or dict keys.
```

## What to add over time

Every week, paste new Q&A pairs under `::instruction::` and run
`dlm train tutor.dlm` again. The delta system (Sprint 08) notices what
changed; prior content stays in the replay corpus so the model doesn't
forget the earlier material.

For code-heavy answers, put the explanation as prose and use
`::instruction::` for conversational questions. Both train, but prose
drives continued pretraining (learns style) while instruction drives
supervised fine-tuning (learns the question → answer mapping).
