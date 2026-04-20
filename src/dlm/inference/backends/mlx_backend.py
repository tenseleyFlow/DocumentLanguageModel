"""MLX inference backend — Apple Silicon only.

`MlxBackend` stages a PEFT adapter into an MLX-loadable directory on
first load and drives generation through `mlx_lm`. Training stays on
PyTorch MPS; this backend is prompt-only.

Staging (audit-08 B3): `mlx_lm.load(adapter_path=<dir>)` expects
`adapters.safetensors` or `adapters.npz` AND `adapter_config.json`
in the same directory. The PEFT source has `adapter_config.json`
plus `adapter_model.safetensors` with wrapper-prefixed tensor keys
(`base_model.model.…`) that don't match MLX-LM's layout. We:

1. Copy `adapter_config.json` verbatim from the PEFT adapter dir.
2. Write `adapters.npz` with the renamed keys (via
   `peft_safetensors_to_mlx_npz`). When `mlx` is importable we prefer
   `mx.savez` to emit the MLX-native layout; otherwise fall through
   to numpy's `.npz` which mlx-lm's loader accepts identically.

Heavy imports (`mlx`, `mlx_lm`) happen inside `load()`. The module
imports only at call time so machines without the `mlx` extra never
trigger a failed import during `dlm` CLI boot.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.inference.backends.base import InferenceBackend
from dlm.inference.errors import AdapterNotFoundError
from dlm.inference.mlx_adapter import MlxConversionError

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.store.paths import StorePath


_ADAPTER_CONFIG_FILENAME = "adapter_config.json"
"""Mlx-lm requires this file alongside the tensor artifact."""


def stage_mlx_adapter_dir(peft_adapter_dir: Path, dst_dir: Path) -> Path:
    """Stage a PEFT adapter dir into an mlx-lm-loadable scratch dir.

    Pure-ish: does not import mlx. Reads PEFT source files, writes a
    minimal dir with `adapters.npz` + `adapter_config.json`.

    Refuses with `MlxConversionError` on anything that isn't a LoRA
    PEFT adapter (audit-08 N10 preflight). Returns the staged dir
    (same as `dst_dir` after writes).
    """
    src_config = peft_adapter_dir / _ADAPTER_CONFIG_FILENAME
    if not src_config.exists():
        raise MlxConversionError(
            f"{peft_adapter_dir} is not a PEFT adapter dir "
            f"({_ADAPTER_CONFIG_FILENAME} is missing)"
        )
    if not (peft_adapter_dir / "adapter_model.safetensors").exists():
        raise MlxConversionError(
            f"{peft_adapter_dir} has no adapter_model.safetensors — "
            "MLX backend only supports LoRA adapters, not merged bases"
        )

    from dlm.inference.mlx_adapter import peft_safetensors_to_mlx_npz

    dst_dir.mkdir(parents=True, exist_ok=True)
    # Copy the config first so a downstream reader sees a consistent
    # dir even if the converter raises mid-write.
    shutil.copy2(src_config, dst_dir / _ADAPTER_CONFIG_FILENAME)
    peft_safetensors_to_mlx_npz(peft_adapter_dir, dst_dir / "adapters.npz")
    return dst_dir


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

        adapter_path = resolve_adapter_path(store, adapter_name=adapter_name)
        if not adapter_path.exists():
            raise AdapterNotFoundError(f"mlx backend: adapter dir {adapter_path} does not exist")

        # Stage both tensors + adapter_config.json into a scratch dir.
        # `stage_mlx_adapter_dir` performs the preflight PEFT-shape
        # check before trying to convert (audit-08 B3 + N10).
        self._workdir = tempfile.TemporaryDirectory(prefix="dlm-mlx-")
        staged = stage_mlx_adapter_dir(adapter_path, Path(self._workdir.name))

        self._model, self._tokenizer = load(
            base.hf_id,
            adapter_path=str(staged),
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
        # Audit-08 N4: the drop is by design, but users need to know
        # their `--top-p` on `--backend mlx` is silently ignored.
        # We could warn here but the tradeoff is noise on every
        # generate call; docs/cli/reference.md calls it out instead.
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
