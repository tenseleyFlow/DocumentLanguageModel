"""MLX inference backend — Apple Silicon only.

`MlxBackend` converts the PEFT adapter to an MLX `.npz` on first load
(cached in a tmpdir next to the adapter) and drives generation through
`mlx_lm`. Training stays on PyTorch MPS; this backend is prompt-only.

Heavy imports (`mlx`, `mlx_lm`) happen inside `load()`. The module
imports only at call time so machines without the `mlx` extra never
trigger a failed import during `dlm` CLI boot.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.inference.backends.base import InferenceBackend
from dlm.inference.errors import AdapterNotFoundError

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.store.paths import StorePath


class MlxBackend(InferenceBackend):
    """mlx-lm driven inference path (Apple Silicon)."""

    name = "mlx"

    def __init__(self, caps: Any) -> None:
        self._caps = caps
        self._model: Any = None
        self._tokenizer: Any = None
        self._workdir: tempfile.TemporaryDirectory[str] | None = None

    def load(  # pragma: no cover - heavy path
        self,
        base: BaseModelSpec,
        store: StorePath,
        *,
        adapter_name: str | None = None,
    ) -> None:
        from mlx_lm import load  # type: ignore[import-not-found]

        from dlm.inference.loader import resolve_adapter_path
        from dlm.inference.mlx_adapter import peft_safetensors_to_mlx_npz

        adapter_path = resolve_adapter_path(store, adapter_name=adapter_name)
        if not adapter_path.exists():
            raise AdapterNotFoundError(f"mlx backend: adapter dir {adapter_path} does not exist")

        # Convert the PEFT adapter into an MLX-loadable .npz in a
        # scratch dir. Kept in a TemporaryDirectory so we clean up on
        # `unload()` — the .npz is a cache of the PEFT file, not a
        # long-lived artifact.
        self._workdir = tempfile.TemporaryDirectory(prefix="dlm-mlx-")
        mlx_adapter = Path(self._workdir.name) / "adapters.npz"
        peft_safetensors_to_mlx_npz(adapter_path, mlx_adapter)

        self._model, self._tokenizer = load(
            base.hf_id,
            adapter_path=str(mlx_adapter.parent),
        )

    def generate(self, prompt: str, **gen_kwargs: Any) -> str:  # pragma: no cover - heavy path
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("MlxBackend.generate called before load()")
        from mlx_lm import generate as mlx_generate

        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 256))
        temperature = float(gen_kwargs.get("temperature", 0.0))

        # mlx-lm's generate() signature: (model, tokenizer, prompt,
        #   max_tokens=..., temp=..., ...). Unknown kwargs from the
        # PyTorch shape (top_p, top_k, repetition_penalty) are
        # dropped rather than raised — backends share a kwargs dict.
        out = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temp=temperature,
            verbose=False,
        )
        return str(out)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._workdir is not None:
            self._workdir.cleanup()
            self._workdir = None
