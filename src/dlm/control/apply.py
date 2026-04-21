"""Attach a control vector to a model at inference time.

The apply path is a thin `forward_pre_hook` over one residual-
stream layer. On every forward pass, the hook adds
`strength * vector` to each token's hidden state. No weight
changes, no retraining — the steering is purely a forward-time
perturbation, which is why extraction takes seconds instead of
hours.

Strength semantics: positive pushes toward the `chosen`
distribution the vector was extracted against; negative pushes
away. Typical range is `[-2, 2]`; beyond `±3` the model tends
to collapse into repetition or nonsense.

Usage:
    ```python
    from dlm.control import apply_control, extract_control_vector

    vec = extract_control_vector(chosen, rejected)
    with apply_control(model, vec.direction, layer_index=12, strength=1.5):
        out = model.generate(...)
    ```

The context manager guarantees the hook is removed on exit,
even when the wrapped block raises — we can't leave a stray
hook on the model, because subsequent unrelated forward passes
would silently keep steering.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from dlm.control.errors import ControlApplyError

if TYPE_CHECKING:
    import torch


def _resolve_layer(model: Any, layer_index: int) -> Any:
    """Locate the residual-stream module at `layer_index`.

    HF decoder-only models expose `model.model.layers[i]` (Llama,
    Qwen, SmolLM, Phi — the canonical path). PEFT wraps that under a
    `base_model.model.model.layers[i]` chain (`PeftModel.base_model`
    is a `LoraModel` whose `.model` is the HF model). Rather than
    hard-code two shapes, we walk down repeated `base_model` / `model`
    hops — the first node that exposes a `layers` attribute wins.
    `layer_index` can be negative (`-1` is the last layer).

    Raises `ControlApplyError` when the walker can't find a `layers`
    attribute anywhere down the chain or when the index is out of
    bounds.
    """
    layers = None
    node: Any = model
    # Cap the walk so a pathological graph can't spin forever; real
    # wrappers are at most 2-3 deep (PEFT adds 2, a rare Accelerate
    # wrapper adds 1).
    for _ in range(6):
        layers = getattr(node, "layers", None)
        if layers is not None:
            break
        next_node = getattr(node, "base_model", None) or getattr(node, "model", None)
        if next_node is None or next_node is node:
            break
        node = next_node

    if layers is None:
        raise ControlApplyError(
            "model exposes no `layers` attribute along the "
            "`base_model` / `model` chain — don't know where to "
            "attach the forward hook. Pass a HF decoder-only model "
            "(or a PEFT-wrapped one), or file an issue with the model "
            "class for wiring."
        )
    try:
        return layers[layer_index]
    except (IndexError, TypeError) as exc:
        raise ControlApplyError(
            f"layer_index={layer_index} out of bounds for a {len(layers)}-layer model"
        ) from exc


def _make_hook(vector: torch.Tensor, strength: float) -> Any:
    """Build a `forward_pre_hook` that adds `strength * vector` to inputs.

    The hook receives `(module, args)` where `args[0]` is the
    hidden-state tensor of shape `(batch, seq, hidden_dim)`. We
    broadcast the vector across the `batch` and `seq` axes — the
    same steering direction applies to every token position, which
    is the canonical control-vector interpretation (steer the entire
    generation, not one token).

    Returns the new args tuple with the perturbed hidden state in
    position 0. HF layers accept positional args for the hidden
    state; kwargs flow through untouched.
    """

    def _hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args:
            return args
        hidden = args[0]
        # Move/cast vector to match hidden's device + dtype.
        steer = vector.to(device=hidden.device, dtype=hidden.dtype)
        perturbed = hidden + strength * steer
        return (perturbed, *args[1:])

    return _hook


@contextmanager
def apply_control(
    model: Any,
    vector: np.ndarray,
    *,
    layer_index: int,
    strength: float = 1.0,
) -> Iterator[Any]:
    """Attach `strength * vector` to the residual stream at `layer_index`.

    Yields the model for use inside a `with` block. On exit — whether
    clean or via exception — the forward hook is removed. No
    weights change; the effect is forward-pass-only.

    Raises `ControlApplyError` on shape mismatch or invalid layer
    index. Shape validation happens up front, not inside the hook,
    so a malformed vector fails before any compute burns.

    `vector` is accepted as NumPy (the storage format) and converted
    to torch on demand — dtype matching to the model's hidden state
    happens inside the hook, so a float32 vector can steer a bf16
    model without explicit casting by the caller.
    """
    import torch  # deferred — apply is runtime-only

    if vector.ndim != 1:
        raise ControlApplyError(
            f"control vector must be 1D (hidden_dim,), got shape {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ControlApplyError("control vector contains non-finite values")

    target_layer = _resolve_layer(model, layer_index)
    # Validate vector length against a weight the layer actually
    # owns. Different architectures put the input-projection under
    # different names — try the common ones.
    expected_dim: int | None = None
    for attr in ("self_attn", "attention", "attn"):
        sub = getattr(target_layer, attr, None)
        if sub is None:
            continue
        for proj_attr in ("q_proj", "qkv_proj"):
            proj = getattr(sub, proj_attr, None)
            if proj is None:
                continue
            weight = getattr(proj, "weight", None)
            if weight is None:
                continue
            expected_dim = int(weight.shape[-1])
            break
        if expected_dim is not None:
            break

    if expected_dim is not None and vector.shape[0] != expected_dim:
        raise ControlApplyError(
            f"control vector dim {vector.shape[0]} does not match model "
            f"hidden dim {expected_dim} at layer {layer_index}"
        )

    vec_tensor = torch.from_numpy(np.ascontiguousarray(vector))
    hook = _make_hook(vec_tensor, strength)
    handle = target_layer.register_forward_pre_hook(hook)
    try:
        yield model
    finally:
        handle.remove()
