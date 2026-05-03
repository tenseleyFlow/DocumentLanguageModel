"""PEFT safetensors → MLX-LM LoRA-adapter converter.

Ships MLX as a second inference backend on Apple Silicon. PEFT writes
LoRA weights as `adapter_model.safetensors` with keys like:

    base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight

`mlx-lm`'s `load_adapters` expects the flattened, lowercased layout:

    model.layers.0.self_attn.q_proj.lora_a
    model.layers.0.self_attn.q_proj.lora_b

(no `base_model` prefix, lowercase `lora_a`/`lora_b`, `.weight` stripped
because mlx-lm adapter files store bare tensors under the final segment).

**Output format: `adapters.safetensors`.** Current mlx-lm
(`tuner/utils.py:137`) hardcodes `model.load_weights(adapters.safetensors)`
— the `.npz` path worked on earlier mlx_lm releases but no longer.
We write safetensors directly. Conversion itself doesn't require MLX
to be importable — the safetensors file is written via the pure
`safetensors.torch` dependency.

**mlx-lm `adapter_config.json` schema.** mlx-lm's loader builds a
`types.SimpleNamespace` from the file and reads `config.num_layers` +
`config.lora_parameters`. PEFT's config has neither. `build_mlx_adapter_config`
translates PEFT-shape into mlx-lm-shape using the HF base config's
`num_hidden_layers` + PEFT's `r` / `lora_alpha` / `lora_dropout` /
`target_modules`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_PEFT_PREFIX = re.compile(r"^base_model\.model\.")
"""PEFT wraps the HF base in a double `base_model.model.` — the outer
`base_model` is the PEFT wrapper, the inner `model` is the HF model
attribute. We strip only the outer `base_model.model.` once so inner
`model.` (HF's actual attribute name) remains."""

_LORA_AB = re.compile(r"\.lora_([AB])\.weight$")
"""Matches the trailing `.lora_A.weight` / `.lora_B.weight` suffix."""

_ATTN_TARGETS: frozenset[str] = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "wqkv"}
)
"""Bare PEFT `target_modules` names that live under `self_attn.` on
decoder-only transformers (Qwen2/Llama/Mistral/Phi/SmolLM)."""

_MLP_TARGETS: frozenset[str] = frozenset({"gate_proj", "up_proj", "down_proj", "fc1", "fc2"})
"""Bare PEFT `target_modules` names that live under `mlp.` on the same
family of architectures."""


def _qualify_target_module(name: str) -> str:
    """Map a PEFT bare `target_modules` entry to its in-block FQN.

    mlx-lm's `linear_to_lora_layers` matches `named_modules()` keys
    *within* each transformer block via exact equality (`if k in keys`).
    PEFT records `target_modules` as bare module names (`q_proj`),
    while the FQN within an MLX-LM transformer block is fully qualified
    (`self_attn.q_proj`). Without this rewrite the keys never match and
    `linear_to_lora_layers` silently leaves the model un-wrapped — the
    user-visible failure is "trained model behaves identically to base."

    Already-qualified names (containing a `.`) pass through untouched
    so callers can pre-qualify if needed.
    """
    if "." in name:
        return name
    if name in _ATTN_TARGETS:
        return f"self_attn.{name}"
    if name in _MLP_TARGETS:
        return f"mlp.{name}"
    return name


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


def peft_safetensors_to_mlx_safetensors(  # pragma: no cover - I/O + torch deps
    peft_adapter_dir: Path,
    mlx_safetensors_path: Path,
) -> dict[str, str]:
    """Convert `<adapter>/adapter_model.safetensors` → `<mlx_safetensors_path>`.

    mlx-lm's current loader reads `adapters.safetensors` (not `.npz`);
    we write safetensors with MLX-shaped keys so `model.load_weights`
    accepts the file without further translation.

    Returns the key mapping actually written (peft_key → mlx_key) so
    callers can log it when `--verbose`.

    Pragma'd: exercised end-to-end by the slow parity integration test
    (covered via `map_all_keys` unit tests for the logic).
    """
    from safetensors.torch import load_file, save_file

    src = peft_adapter_dir / "adapter_model.safetensors"
    if not src.exists():
        raise MlxConversionError(f"no adapter_model.safetensors in {peft_adapter_dir}")

    tensors = load_file(str(src))
    mapping = map_all_keys(list(tensors.keys()))

    # PEFT stores LoRA weights with shapes that don't match what
    # mlx-lm's `LoRALinear` expects:
    #
    #   PEFT lora_A : [r, in_features]       MLX lora_a : [in_features, r]
    #   PEFT lora_B : [out_features, r]      MLX lora_b : [r, out_features]
    #
    # Both tensors need a transpose. Loading without the transpose
    # makes mlx-lm's `model.load_weights(strict=False)` silently skip
    # the mismatched shapes and the adapter has no effect — the
    # textbook "trained model behaves like base" failure mode.
    mlx_tensors = {}
    for peft_key, mlx_key in mapping.items():
        t = tensors[peft_key]
        # `mlx_key` ends in `.lora_a` or `.lora_b` (lowercase, no `.weight`).
        if mlx_key.endswith((".lora_a", ".lora_b")):
            t = t.t().contiguous()
        mlx_tensors[mlx_key] = t

    mlx_safetensors_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(mlx_tensors, str(mlx_safetensors_path))
    return mapping


def build_mlx_adapter_config(
    peft_config: dict[str, Any],
    base_num_hidden_layers: int,
) -> dict[str, Any]:
    """Translate a PEFT `adapter_config.json` into mlx-lm's schema.

    mlx-lm's `load_adapters` (tuner/utils.py) reads:

    - `config.num_layers` — how many trailing layers receive LoRA
      (matches mlx-lm convention: -1 = all; a positive N = last N).
      We emit the base model's `num_hidden_layers` so every layer
      gets the adapter — matches PEFT default when `layers_to_transform`
      isn't set.
    - `config.lora_parameters.rank` ← PEFT `r`
    - `config.lora_parameters.scale` ← PEFT `lora_alpha / r`
    - `config.lora_parameters.dropout` ← PEFT `lora_dropout`
    - `config.lora_parameters.keys` ← PEFT `target_modules`
    - `config.fine_tune_type` — "lora" unless PEFT `use_dora=True`,
      in which case "dora".

    Fails loud (`MlxConversionError`) when the PEFT config is missing
    fields we cannot substitute — `r` and `target_modules` in particular
    are load-bearing.
    """
    try:
        rank = int(peft_config["r"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MlxConversionError(
            f"PEFT adapter_config.json missing or non-integer 'r' (LoRA rank): {exc}"
        ) from exc
    target_modules = peft_config.get("target_modules")
    if not isinstance(target_modules, list) or not target_modules:
        raise MlxConversionError(
            "PEFT adapter_config.json 'target_modules' must be a non-empty list; got "
            f"{target_modules!r}. mlx-lm needs this to wire LoRA into the right ops."
        )
    lora_alpha = float(peft_config.get("lora_alpha", rank))
    lora_dropout = float(peft_config.get("lora_dropout", 0.0))
    use_dora = bool(peft_config.get("use_dora", False))

    if base_num_hidden_layers < 1:
        raise MlxConversionError(
            f"base model reports num_hidden_layers={base_num_hidden_layers} (expected >=1); "
            "cannot stage mlx adapter without a valid layer count"
        )

    qualified_keys = [_qualify_target_module(t) for t in target_modules]

    return {
        "fine_tune_type": "dora" if use_dora else "lora",
        "num_layers": int(base_num_hidden_layers),
        "lora_parameters": {
            "rank": rank,
            "scale": lora_alpha / rank if rank else float(lora_alpha),
            "dropout": lora_dropout,
            "keys": qualified_keys,
        },
    }


def assert_mlx_adapter_applied(model: Any, *, expected_keys: list[str]) -> None:
    """Verify mlx-lm's `load_adapters` actually wrapped the targeted layers.

    `mlx_lm.load(..., adapter_path=...)` calls `linear_to_lora_layers`
    followed by `model.load_weights(strict=False)`. Both steps fail
    silently if their inputs don't match the loaded model:

    - `linear_to_lora_layers` is a no-op when `keys` don't match any
      module's FQN inside the transformer blocks
    - `load_weights(strict=False)` skips any tensor key that doesn't
      match a model parameter

    Either failure produces a model that runs as if no adapter were
    loaded. Catching this here turns the "trained model behaves like
    base" footgun into an explicit refusal so the user knows to use
    `--backend pytorch` (or the fix needs an architecture-aware
    keys translator).

    `expected_keys` are the in-block FQNs from the staged
    `adapter_config.json` (e.g. `["self_attn.q_proj", ...]`). We confirm
    that at least one matching module ended up as a LoRA-wrapped layer.
    """
    try:
        import mlx.utils as mlx_utils  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:  # pragma: no cover - mlx not importable
        raise MlxConversionError(f"mlx not importable for verification: {exc}") from exc

    try:
        flat: Any = mlx_utils.tree_flatten(model.trainable_parameters())
    except Exception as exc:  # pragma: no cover - defensive
        raise MlxConversionError(
            f"could not enumerate model trainable_parameters for verification: {exc}"
        ) from exc

    lora_param_count = sum(1 for k, _ in flat if k.endswith(".lora_a") or k.endswith(".lora_b"))
    if lora_param_count == 0:
        raise MlxConversionError(
            "mlx-lm loaded the adapter without applying it — zero "
            "`lora_a` / `lora_b` parameters present after load. This "
            "usually means the keys "
            f"{expected_keys!r} don't match the model's `named_modules()` "
            "FQNs (e.g. the base architecture uses a different submodule "
            "layout than `self_attn.*` / `mlp.*`). The trained adapter "
            "would behave identically to the base model. Use "
            "`--backend pytorch` as a workaround."
        )
