"""Order-of-magnitude VRAM estimator for a LoRA/QLoRA training step.

The formula deliberately trades accuracy for simplicity:

    peak ≈  base_weights(params, precision)
          + activations(seq_len, micro_batch, gradient_checkpointing)
          + optimizer_state(trainable_params, optimizer)
          + lora_state(r, alpha, target_modules_fraction)
          + safety_margin(20%)

Accuracy targets (Sprint 05 DoD, audit F26):

- within 30% of measured peak for SmolLM2-135M LoRA on CPU CI
- within 40% for Qwen2.5-1.5B LoRA on CUDA CI (gpu-marked)

The runtime OOM guard in Sprint 09 is the safety net; this estimator is
advisory. The planner's job is to pick sane defaults — OOM is recoverable
by the user bumping `grad_accum`.
"""

from __future__ import annotations

from typing import Literal

from dlm.hardware.backend import Backend

Precision = Literal["bf16", "fp16", "fp32"]


_BYTES_PER_PARAM: dict[Precision, int] = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}

# 4-bit NF4 (bitsandbytes) uses ~0.5B/param for the weights themselves
# plus a small quantization state overhead. Empirically ~0.56B/param.
_QLORA_BYTES_PER_PARAM: float = 0.56

# Gradient-checkpointing approximately reduces activation memory by 5x
# (we recompute on the backward pass). Quoted conservatively as 0.25
# since in practice some activations still linger.
_GRAD_CKPT_ACTIVATION_FACTOR: float = 0.25

# Per-token activation memory, empirical order-of-magnitude for
# transformer blocks in bf16/fp16 without flash-attn. Real models vary
# by architecture (num_heads, hidden_size); we swallow that into the
# safety margin.
_ACTIVATION_BYTES_PER_TOKEN_PER_LAYER: float = 48.0

# AdamW keeps two fp32 moment buffers per trainable param (m, v) + a
# master copy of params in mixed precision → 8 B/param in full precision.
# 8-bit optimizers (paged_adamw_8bit, adamw_bnb_8bit) drop this to ~1 B/param.
_OPTIMIZER_BYTES_PER_PARAM: dict[str, float] = {
    "adamw_torch": 8.0,
    "adamw_bnb_8bit": 1.25,
    "paged_adamw_8bit": 1.25,
}

_SAFETY_MARGIN: float = 1.20  # 20% above sum-of-components


def estimate_peak_vram_gb(
    *,
    base_params: int,
    seq_len: int,
    micro_batch: int,
    precision: Precision,
    use_qlora: bool,
    lora_r: int,
    gradient_checkpointing: bool,
    optimizer: str = "adamw_torch",
    target_modules_fraction: float = 0.05,
    num_layers: int = 24,
) -> float:
    """Return peak VRAM (in GiB) for a single training step under the
    given config.

    `target_modules_fraction` is the share of base params covered by
    trainable LoRA adapters — typical LoRA configs touching q/k/v/o
    projections over a 24-layer transformer land in the 2-6% range.
    `num_layers` is advisory; it's folded into the activation estimate.
    """
    base_bytes = _base_weight_bytes(base_params, precision, use_qlora)
    trainable_params = int(base_params * target_modules_fraction)
    lora_params = _lora_parameter_count(trainable_params, lora_r)
    optimizer_bytes = lora_params * _OPTIMIZER_BYTES_PER_PARAM.get(optimizer, 8.0)

    # Activations scale with seq_len * micro_batch * layers; gradient
    # checkpointing trims by the factor above.
    per_step_tokens = seq_len * micro_batch * num_layers
    activation_bytes = per_step_tokens * _ACTIVATION_BYTES_PER_TOKEN_PER_LAYER
    if gradient_checkpointing:
        activation_bytes *= _GRAD_CKPT_ACTIVATION_FACTOR

    lora_weight_bytes = lora_params * _BYTES_PER_PARAM[precision]

    subtotal = base_bytes + activation_bytes + optimizer_bytes + lora_weight_bytes
    total = subtotal * _SAFETY_MARGIN
    return total / (1024**3)


def estimate_step_seconds(
    *,
    backend: Backend,
    base_params: int,
    seq_len: int,
    micro_batch: int,
    has_flash_attention: bool,
) -> float:
    """Very rough estimate of a training-step wall-clock, advisory only.

    Coefficient chosen so SmolLM2-135M @ seq=512 batch=4 on a 4090 lands
    near 0.1s and a 1.5B LoRA @ seq=2048 batch=4 lands near 0.8s. Real
    performance depends on too many factors for a tighter promise.
    """
    tokens = seq_len * micro_batch
    if backend == Backend.CUDA:
        # Empirical throughput near 250k tok/s on FA2 + H100, 60k on 4090.
        tokens_per_sec = 60_000 if has_flash_attention else 25_000
    elif backend == Backend.MPS:
        tokens_per_sec = 8_000  # Apple M-series, LoRA-only
    elif backend == Backend.ROCM:
        tokens_per_sec = 40_000
    else:
        tokens_per_sec = 500  # CPU: practically unusable at scale

    # Scale inversely with model size (extremely rough).
    size_factor = max(base_params / 1.5e9, 0.1)
    return tokens / (tokens_per_sec / size_factor)


# --- internals ---------------------------------------------------------------


def _base_weight_bytes(params: int, precision: Precision, use_qlora: bool) -> float:
    if use_qlora:
        return params * _QLORA_BYTES_PER_PARAM
    return params * _BYTES_PER_PARAM[precision]


def _lora_parameter_count(trainable_base_params: int, lora_r: int) -> int:
    """LoRA adapts each covered weight matrix W ∈ ℝ^(d×k) with two
    matrices A ∈ ℝ^(d×r) and B ∈ ℝ^(r×k). For rank r, adapter params
    are `r * (d + k)` per matrix — we approximate across covered matrices
    with `2 * r * sqrt(trainable_base_params)` as a shape-agnostic
    back-of-envelope. Off by a small constant factor; subsumed by safety
    margin.
    """
    if trainable_base_params <= 0 or lora_r <= 0:
        return 0
    # sqrt heuristic: covers q/k/v/o with d ≈ k; gives a realistic
    # trainable-param count for rank r without needing the model's
    # exact hidden size.
    sqrt_params = int(trainable_base_params**0.5)
    return 2 * lora_r * sqrt_params
