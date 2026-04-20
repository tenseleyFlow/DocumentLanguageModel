"""PEFT safetensors → MLX-LM `.npz` LoRA-adapter converter.

Sprint 21 ships MLX as a second inference backend on Apple Silicon.
PEFT writes LoRA weights as `adapter_model.safetensors` with keys like:

    base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight

`mlx-lm`'s `load_adapters` expects the flattened, lowercased layout:

    model.layers.0.self_attn.q_proj.lora_a
    model.layers.0.self_attn.q_proj.lora_b

(no `base_model` prefix, lowercase `lora_a`/`lora_b`, `.weight` stripped
because mlx-lm adapter files store bare tensors under the final segment).

The converter is split so the key-mapping logic is pure and
unit-testable; the tensor I/O layer is a thin wrapper around
`safetensors.torch.load_file` + `numpy.savez`. We write `.npz` via
numpy (not `mx.savez`) so conversion itself does not require MLX to
be importable — the artifact is still MLX-loadable because `mx.load`
understands numpy's `.npz` format.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


_PEFT_PREFIX = re.compile(r"^base_model\.model\.")
"""PEFT wraps the HF base in a double `base_model.model.` — the outer
`base_model` is the PEFT wrapper, the inner `model` is the HF model
attribute. We strip only the outer `base_model.model.` once so inner
`model.` (HF's actual attribute name) remains."""

_LORA_AB = re.compile(r"\.lora_([AB])\.weight$")
"""Matches the trailing `.lora_A.weight` / `.lora_B.weight` suffix."""


class MlxConversionError(RuntimeError):
    """Raised when a PEFT adapter cannot be converted to the MLX layout."""


def map_peft_key_to_mlx(peft_key: str) -> str | None:
    """Return the MLX-LM key for a PEFT tensor, or None if the key should be skipped.

    Rules:
    - Strip the leading `base_model.model.` wrapper prefix. PEFT's
      outer wrapper is redundant under MLX-LM's flattened naming.
    - Rewrite `...lora_A.weight` → `...lora_a` and `...lora_B.weight`
      → `...lora_b`. Case change + suffix drop in one pass.
    - Return None for any tensor that isn't a LoRA A/B pair (e.g.
      `modules_to_save` copies of `embed_tokens` — MLX-LM handles those
      via the base model, not the adapter file).

    Pure string transformation; no tensor shape changes happen here.
    """
    if not _LORA_AB.search(peft_key):
        return None
    stripped = _PEFT_PREFIX.sub("", peft_key, count=1)
    return _LORA_AB.sub(lambda m: f".lora_{m.group(1).lower()}", stripped)


def map_all_keys(peft_keys: list[str]) -> dict[str, str]:
    """Build the peft_key → mlx_key mapping for a whole adapter file.

    Non-LoRA keys are silently dropped (see `map_peft_key_to_mlx`).
    Duplicate output keys trigger `MlxConversionError` — that would
    mean two PEFT tensors collapsed to the same MLX name, which
    silently overwriting would mask a real adapter-layout bug.
    """
    mapping: dict[str, str] = {}
    seen: dict[str, str] = {}
    for key in peft_keys:
        mapped = map_peft_key_to_mlx(key)
        if mapped is None:
            continue
        if mapped in seen:
            raise MlxConversionError(
                f"two PEFT keys map to the same MLX key {mapped!r}: {seen[mapped]!r} and {key!r}"
            )
        seen[mapped] = key
        mapping[key] = mapped
    if not mapping:
        raise MlxConversionError(
            "PEFT adapter has no LoRA A/B weight tensors — not a convertible LoRA checkpoint"
        )
    return mapping


def peft_safetensors_to_mlx_npz(  # pragma: no cover - I/O + torch deps
    peft_adapter_dir: Path,
    mlx_npz_path: Path,
) -> dict[str, str]:
    """Convert `<adapter>/adapter_model.safetensors` → `<mlx_npz_path>`.

    Returns the key mapping actually written (peft_key → mlx_key) so
    callers can log it when `--verbose`.

    Pragma'd: exercised end-to-end by the slow parity integration test
    (covered via `map_all_keys` unit tests for the logic).
    """
    import numpy as np
    from safetensors.torch import load_file

    src = peft_adapter_dir / "adapter_model.safetensors"
    if not src.exists():
        raise MlxConversionError(f"no adapter_model.safetensors in {peft_adapter_dir}")

    tensors = load_file(str(src))
    mapping = map_all_keys(list(tensors.keys()))

    np_tensors: dict[str, NDArray[np.float32]] = {}
    for peft_key, mlx_key in mapping.items():
        tensor = tensors[peft_key]
        # safetensors.torch.load_file returns torch.Tensor; .numpy()
        # is the standard bridge. fp16 is preserved across the write.
        np_tensors[mlx_key] = tensor.detach().cpu().numpy()

    mlx_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(mlx_npz_path), **np_tensors)  # type: ignore[arg-type]
    return mapping
