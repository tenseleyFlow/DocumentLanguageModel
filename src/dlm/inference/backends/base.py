"""`InferenceBackend` Protocol shared by PyTorch + MLX paths.

MLX provides a second inference backend for Apple Silicon throughput.
The existing PyTorch path stays authoritative on every other platform
and remains the training-time runtime. This Protocol is the shape both
paths satisfy so the CLI + REPL can treat them interchangeably.

Backends are stateful: `load()` resolves the adapter, loads weights,
and stashes the live model on `self`; `generate()` is called repeatedly
against that loaded state; `unload()` releases memory. Pooling /
reuse across CLI invocations is a later concern — the shape supports
it without mandating it yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.store.paths import StorePath


@runtime_checkable
class InferenceBackend(Protocol):
    """Minimal inference-backend interface.

    Implementations live under `dlm.inference.backends.<name>_backend`.
    `runtime_checkable` so `isinstance(obj, InferenceBackend)` works in
    tests without requiring a real concrete class.
    """

    name: str
    """Short stable identifier — `"pytorch"` or `"mlx"`. Reported in
    `--verbose` output and error messages."""

    def load(
        self,
        base: BaseModelSpec,
        store: StorePath,
        *,
        adapter_name: str | None = None,
    ) -> None:
        """Resolve + load base + adapter + tokenizer into instance state.

        `adapter_name=None` selects the flat single-adapter layout;
        a non-None name selects `adapter/<name>/` under the named
        multi-adapter layout. Raises `AdapterNotFoundError` when the
        pointer is missing (delegated to the concrete backend).
        """
        ...

    def generate(self, prompt: str, **gen_kwargs: Any) -> str:
        """Render `prompt`, run generation, return the response string.

        `gen_kwargs` are backend-agnostic (max_new_tokens, temperature,
        top_p, top_k, repetition_penalty). Each backend maps these to
        its native generation API; unknown kwargs are ignored, not
        raised, so a caller can pass the PyTorch-shaped kwargs through
        to MLX without pre-filtering.
        """
        ...

    def unload(self) -> None:
        """Release loaded weights + tokenizer. Safe to call repeatedly."""
        ...
