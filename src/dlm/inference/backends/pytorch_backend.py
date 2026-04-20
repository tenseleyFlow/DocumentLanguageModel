"""PyTorch / HuggingFace inference backend.

Wraps the existing `load_for_inference` + `generate` surface behind the
`InferenceBackend` Protocol. No behavior change for current callers —
the module-level helpers remain the canonical single-shot entry points;
this backend is the stateful, REPL-friendly twin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dlm.inference.backends.base import InferenceBackend

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.inference.loader import LoadedInference
    from dlm.store.paths import StorePath


class PyTorchBackend(InferenceBackend):
    """HF + PEFT + torch inference path (the v1.0 default)."""

    name = "pytorch"

    def __init__(self, caps: Any) -> None:
        self._caps = caps
        self._loaded: LoadedInference | None = None

    def load(
        self,
        base: BaseModelSpec,
        store: StorePath,
        *,
        adapter_name: str | None = None,
    ) -> None:
        from dlm.inference.loader import load_for_inference

        self._loaded = load_for_inference(
            store, base, self._caps, adapter_name=adapter_name
        )

    def generate(self, prompt: str, **gen_kwargs: Any) -> str:
        if self._loaded is None:
            raise RuntimeError("PyTorchBackend.generate called before load()")
        from dlm.inference.generate import generate as _generate

        return _generate(
            self._loaded.model,
            self._loaded.tokenizer,
            prompt,
            **gen_kwargs,
        )

    def unload(self) -> None:
        self._loaded = None
