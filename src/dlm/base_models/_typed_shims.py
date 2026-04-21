"""Typed wrappers around transformers classes whose stubs are untyped.

Mypy --strict flags `AutoProcessor.from_pretrained` as a call to an
untyped function because the transformers type stubs leave the method
typed `-> Any` on a `@classmethod` that lands after a Union resolution
mypy can't follow. We call it from six call sites; centralizing the
`cast(Any, ...)` here beats sprinkling `# type: ignore` across the
tree (CLAUDE.md contract: "never loosen; fix the type at source").

Each shim preserves the original call shape (kwargs passthrough) and
returns `Any` — the cost of a silent API change upstream is already
paid by the runtime probe suite; we don't gain safety from a narrower
return type here.
"""

from __future__ import annotations

from typing import Any


def load_auto_processor(hf_id: str, **kwargs: Any) -> Any:
    """`transformers.AutoProcessor.from_pretrained(hf_id, **kwargs)`.

    Centralized so `mypy --strict` sees one well-typed call instead of
    six. Callers handle the `Any` result the same way they would have
    handled the raw `from_pretrained` return — the processor is used
    as an opaque handle (passed to `processor(...)` + `.tokenizer`
    attr access).
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(hf_id, **kwargs)  # type: ignore[no-untyped-call]
