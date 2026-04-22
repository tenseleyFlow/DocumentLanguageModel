"""MLX inference backend — Apple Silicon only.

`MlxBackend` stages a PEFT adapter into an MLX-loadable directory on
first load and drives generation through `mlx_lm`. Training stays on
PyTorch MPS; this backend is prompt-only.

Staging: `mlx_lm.load(adapter_path=<dir>)` expects
`adapters.safetensors` AND `adapter_config.json` in that directory,
and mlx-lm's loader reads `config.num_layers` + `config.lora_parameters`
off a `SimpleNamespace` built from the config file. PEFT's
`adapter_config.json` has neither field — PEFT uses `r`, `lora_alpha`,
`target_modules`, etc. A verbatim copy (the prior approach) crashes
with `AttributeError: 'types.SimpleNamespace' object has no attribute
'num_layers'` on the first prompt. We:

1. Read the PEFT `adapter_config.json` + the HF base config to learn
   `num_hidden_layers`.
2. Write an mlx-lm-shaped `adapter_config.json` via
   `build_mlx_adapter_config` (rank, scale, dropout, keys, num_layers).
3. Write `adapters.safetensors` with MLX-shaped keys via
   `peft_safetensors_to_mlx_safetensors`. Current mlx-lm hardcodes
   `.safetensors`; the earlier `.npz` path no longer loads.

Heavy imports (`mlx`, `mlx_lm`) happen inside `load()`. The module
imports only at call time so machines without the `mlx` extra never
trigger a failed import during `dlm` CLI boot.
"""

from __future__ import annotations

import json
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


def _resolve_base_num_hidden_layers(base_hf_id: str) -> int:
    """Look up the HF base model's `num_hidden_layers` from the local cache.

    Prefers `transformers.AutoConfig.from_pretrained(base_hf_id,
    local_files_only=True)` so nothing goes over the network at
    prompt time. Falls back to reading the raw `config.json` from the
    HF cache snapshot directory when `transformers` isn't importable
    (e.g., minimal-install inference).

    `MlxConversionError` on miss — we need this field to emit a usable
    mlx-lm adapter config.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(base_hf_id, local_files_only=True)
        num_layers = getattr(cfg, "num_hidden_layers", None)
        if isinstance(num_layers, int) and num_layers > 0:
            return num_layers
    except Exception:  # noqa: BLE001 — transformers optional; fall through to cache read
        pass

    # Fallback: HF cache holds config.json at
    # ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<sha>/config.json
    try:
        from huggingface_hub import snapshot_download

        snapshot_dir = Path(
            snapshot_download(
                repo_id=base_hf_id, local_files_only=True, allow_patterns=["config.json"]
            )
        )
        cfg_path = snapshot_dir / "config.json"
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            num_layers = data.get("num_hidden_layers")
            if isinstance(num_layers, int) and num_layers > 0:
                return num_layers
    except Exception as exc:  # noqa: BLE001 — translate to typed error below
        raise MlxConversionError(
            f"could not resolve num_hidden_layers for {base_hf_id!r} "
            f"from local HF cache ({type(exc).__name__}: {exc})"
        ) from exc

    raise MlxConversionError(
        f"base model {base_hf_id!r} has no usable num_hidden_layers in the local HF cache"
    )


def stage_mlx_adapter_dir(
    peft_adapter_dir: Path,
    dst_dir: Path,
    *,
    base_hf_id: str,
) -> Path:
    """Stage a PEFT adapter dir into an mlx-lm-loadable scratch dir.

    Reads PEFT source files + the HF base model's `num_hidden_layers`,
    writes `adapters.safetensors` (mlx-shaped keys) + an mlx-lm-shape
    `adapter_config.json` with `num_layers` / `lora_parameters` so
    mlx-lm's `load_adapters` accepts the dir.

    Refuses with `MlxConversionError` on anything that isn't a LoRA
    PEFT adapter. Returns the staged dir (same as `dst_dir`).
    """
    src_config = peft_adapter_dir / _ADAPTER_CONFIG_FILENAME
    if not src_config.exists():
        raise MlxConversionError(
            f"{peft_adapter_dir} is not a PEFT adapter dir ({_ADAPTER_CONFIG_FILENAME} is missing)"
        )
    if not (peft_adapter_dir / "adapter_model.safetensors").exists():
        raise MlxConversionError(
            f"{peft_adapter_dir} has no adapter_model.safetensors — "
            "MLX backend only supports LoRA adapters, not merged bases"
        )

    try:
        peft_config = json.loads(src_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MlxConversionError(f"cannot read {src_config}: {type(exc).__name__}: {exc}") from exc

    from dlm.inference.mlx_adapter import (
        build_mlx_adapter_config,
        peft_safetensors_to_mlx_safetensors,
    )

    base_layers = _resolve_base_num_hidden_layers(base_hf_id)
    mlx_config = build_mlx_adapter_config(peft_config, base_layers)

    dst_dir.mkdir(parents=True, exist_ok=True)
    # Write the translated config first so a downstream reader sees a
    # consistent dir even if the tensor converter raises mid-write.
    (dst_dir / _ADAPTER_CONFIG_FILENAME).write_text(
        json.dumps(mlx_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    peft_safetensors_to_mlx_safetensors(peft_adapter_dir, dst_dir / "adapters.safetensors")
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
        from mlx_lm import load  # type: ignore[import-not-found, unused-ignore]

        from dlm.inference.loader import resolve_adapter_path

        adapter_path = resolve_adapter_path(store, adapter_name=adapter_name)
        if not adapter_path.exists():
            raise AdapterNotFoundError(f"mlx backend: adapter dir {adapter_path} does not exist")

        # Stage both tensors + adapter_config.json into a scratch dir.
        # `stage_mlx_adapter_dir` performs the preflight PEFT-shape
        # check + translates PEFT config into mlx-lm's schema + writes
        # adapters.safetensors (mlx-lm's current API reads .safetensors,
        # not .npz, so a verbatim copy of the PEFT config + .npz weights
        # crashes with AttributeError: 'num_layers').
        self._workdir = tempfile.TemporaryDirectory(prefix="dlm-mlx-")
        staged = stage_mlx_adapter_dir(
            adapter_path, Path(self._workdir.name), base_hf_id=base.hf_id
        )

        # mlx_lm.load returns (model, tokenizer) when return_config is
        # False (the default) — `misc` suppresses the Union unpacking
        # warning. `unused-ignore` guards against CI linux where mlx_lm
        # is not importable and this branch isn't analyzed.
        self._model, self._tokenizer = load(  # type: ignore[misc, unused-ignore]
            base.hf_id,
            adapter_path=str(staged),
        )

    def generate(self, prompt: str, **gen_kwargs: Any) -> str:  # pragma: no cover - heavy path
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("MlxBackend.generate called before load()")
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        # kwargs may carry None for unset params (the PyTorch backend
        # treats None as "use default"); coerce to the mlx-sampler
        # neutral values before float/int conversion.
        max_new_tokens = int(gen_kwargs.get("max_new_tokens") or 256)
        temperature = float(gen_kwargs.get("temperature") or 0.0)
        top_p = float(gen_kwargs.get("top_p") or 0.0)
        top_k = int(gen_kwargs.get("top_k") or 0)

        # mlx-lm's `generate` / `generate_step` no longer accept `temp`
        # directly — sampling params are passed via a sampler produced
        # by `make_sampler(temp=..., top_p=..., top_k=...)`. Unknown
        # PyTorch-shape kwargs (repetition_penalty) are dropped;
        # docs/cli/reference.md documents the mlx backend's coverage.
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        out = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        return str(out)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._workdir is not None:
            self._workdir.cleanup()
            self._workdir = None
