"""`InferencePlan` — cross-hardware load plan for prompt-time (audit F05).

The problem
-----------

A QLoRA adapter trained on a CUDA host stores weights that expect
`bitsandbytes` at load time. Move that adapter to a CPU-only laptop
or an Apple Silicon box (no bnb, no CUDA) and a naive `PeftModel.from_pretrained`
either crashes (missing bnb) or silently loads the wrong layout. Audit
F05 flags this as a correctness hazard: the same `.dlm` must produce
coherent `dlm prompt` output on whatever hardware the user happens to
be on.

The solution
------------

`InferencePlan` is the twin of Sprint 05's `TrainingPlan`: a
hardware-doctor decision, but for the inference path. It reads the
saved adapter's training metadata (`pinned_versions.json`) to learn
whether QLoRA was in play, cross-references with the current `Capabilities`,
and emits:

- `dequantize_on_load=True` iff the adapter was QLoRA-trained but the
  current host lacks bnb. The loader then dequantizes to fp16 on
  load via `torch.load` → `model.to(dtype=fp16)` rather than
  `BitsAndBytesConfig(load_in_4bit=True)`.
- `precision` — bf16 on Ampere+ CUDA, fp16 everywhere else.
- `attn_implementation` — sdpa as the safe default; flash_attention_2
  only when the hardware + torch build both support it.

`dlm prompt --verbose` prints the plan so cross-machine workflows are
debuggable without reading source.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from dlm.hardware.backend import Backend

PrecisionLit = Literal["bf16", "fp16"]
AttnImpl = Literal["sdpa", "flash_attention_2", "eager"]


@dataclass(frozen=True)
class InferencePlan:
    """Cross-hardware load plan for `dlm prompt`."""

    backend: Backend
    precision: PrecisionLit
    dequantize_on_load: bool
    attn_implementation: AttnImpl
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Human/JSON-friendly view for `--verbose` output."""
        data = asdict(self)
        data["backend"] = str(self.backend)
        return data


def resolve_inference(adapter_dir: Path, caps: Any) -> InferencePlan:
    """Decide how to load the adapter on the current host.

    `caps` is a `dlm.hardware.Capabilities` snapshot. Kept as `Any` at
    the signature to avoid a runtime import cycle through
    `dlm.hardware` — all accesses are attribute-style and failure-free.

    Decision tree:
    - CUDA host + bnb installed + QLoRA-trained → 4-bit load, no dequant.
    - CUDA host, QLoRA-trained, but bnb missing → dequantize to fp16.
    - Non-CUDA host + QLoRA-trained → dequantize to fp16 (the "audit
      F05" scenario: laptop inference of a server-trained adapter).
    - Non-QLoRA adapter → load at the host's best precision (bf16 on
      capable CUDA, else fp16).
    """
    was_qlora = _was_adapter_qlora(adapter_dir)
    backend = caps.backend

    if backend == Backend.CUDA:
        has_bnb = bool(getattr(caps, "has_bitsandbytes", False))
        if was_qlora and has_bnb:
            return InferencePlan(
                backend=backend,
                precision=_best_cuda_precision(caps),
                dequantize_on_load=False,
                attn_implementation=_pick_attn(caps),
                reason="QLoRA adapter loaded 4-bit via bitsandbytes on CUDA.",
            )
        if was_qlora and not has_bnb:
            return InferencePlan(
                backend=backend,
                precision="fp16",
                dequantize_on_load=True,
                attn_implementation=_pick_attn(caps),
                reason=(
                    "QLoRA adapter but bitsandbytes not installed; dequantizing to fp16 on load."
                ),
            )
        # Plain LoRA on CUDA.
        return InferencePlan(
            backend=backend,
            precision=_best_cuda_precision(caps),
            dequantize_on_load=False,
            attn_implementation=_pick_attn(caps),
            reason="LoRA adapter on CUDA; native dtype load.",
        )

    # MPS / CPU / ROCm — no 4-bit support path. QLoRA adapters require
    # dequantization; plain LoRA just loads at fp16.
    precision: PrecisionLit = "fp16"
    if was_qlora:
        return InferencePlan(
            backend=backend,
            precision=precision,
            dequantize_on_load=True,
            attn_implementation="sdpa",
            reason=(
                f"QLoRA adapter on {backend} host; dequantizing to fp16 "
                "(bitsandbytes is CUDA-only). Audit F05 cross-hardware path."
            ),
        )
    return InferencePlan(
        backend=backend,
        precision=precision,
        dequantize_on_load=False,
        attn_implementation="sdpa",
        reason=f"LoRA adapter on {backend}; fp16 load.",
    )


# --- internals ---------------------------------------------------------------


def _was_adapter_qlora(adapter_dir: Path) -> bool:
    """Read the explicit `use_qlora` flag from the training-run sidecar.

    Audit-05 M1: earlier versions inferred this from the `bitsandbytes`
    package pin, which false-positived on plain LoRA runs on
    CUDA+bnb hosts. The training-state sidecar now writes
    `training_run.json` with an explicit `use_qlora: bool` that records
    what the *training plan* decided, not what happened to be installed.

    Missing or malformed `training_run.json` → fall back to the bnb
    pin heuristic for backwards compatibility with adapters trained
    before Sprint 10's audit-05 patch.
    """
    training_run_path = adapter_dir / "training_run.json"
    if training_run_path.exists():
        try:
            data = json.loads(training_run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        else:
            if "use_qlora" in data:
                return bool(data["use_qlora"])

    # Legacy fallback: pre-audit-05 adapters only have pinned_versions.json.
    # This is imprecise (see audit-05 M1) but preserves behavior for
    # already-written checkpoints that don't have the explicit flag.
    pinned_path = adapter_dir / "pinned_versions.json"
    if not pinned_path.exists():
        return False
    try:
        pinned = json.loads(pinned_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    bnb = pinned.get("bitsandbytes")
    return isinstance(bnb, str) and bool(bnb)


def _best_cuda_precision(caps: Any) -> PrecisionLit:
    if bool(getattr(caps, "supports_bf16", False)):
        return "bf16"
    return "fp16"


def _pick_attn(caps: Any) -> AttnImpl:
    if bool(getattr(caps, "has_flash_attention", False)):
        return "flash_attention_2"
    return "sdpa"
