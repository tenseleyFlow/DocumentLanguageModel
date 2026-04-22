"""Shared QLoRA precision-safety rules for export and inference.

Two concerns live here:

1. Read adapter metadata to decide whether training actually used QLoRA.
2. Enforce the merged-export dequantize gate for QLoRA adapters.

This keeps the `training_run.json` -> `pinned_versions.json` fallback in
one place so export and inference do not drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlm.export.errors import PreflightError, UnsafeMergeError

_UNSAFE_MERGE_MESSAGE = (
    "This adapter was trained on a 4-bit base (QLoRA). "
    "Merging loses precision silently.\n"
    "Re-run with `--merged --dequantize` to proceed in fp16, "
    "or drop `--merged` to use the default separate-GGUF path."
)


@dataclass(frozen=True)
class PrecisionSafetyDecision:
    """Export-side merge-safety verdict for one adapter."""

    was_qlora: bool
    safe: bool
    reason: str
    requires_dequantize: bool


def was_trained_with_qlora(
    adapter_dir: Path,
    *,
    strict_training_run: bool = False,
) -> bool:
    """Return True iff adapter metadata says training used QLoRA.

    Preferred source is `training_run.json`'s explicit `use_qlora` flag.
    Missing or legacy metadata falls back to the older
    `pinned_versions.json` / `bitsandbytes` heuristic for backwards
    compatibility.

    `strict_training_run=True` is the export-time policy: if an existing
    `training_run.json` is unreadable, raise instead of silently weakening
    the merge-safety gate. Inference uses the permissive default so older
    or partially-migrated adapters can still load.
    """
    training_run_path = adapter_dir / "training_run.json"
    if training_run_path.exists():
        try:
            data = json.loads(training_run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if strict_training_run:
                raise PreflightError(
                    probe="training_run_json",
                    detail=(
                        f"adapter {adapter_dir}/training_run.json is unreadable "
                        f"({exc}); cannot determine use_qlora, refusing merge-safety "
                        "bypass. Re-run `dlm train` or fix the file."
                    ),
                ) from exc
        else:
            flag = data.get("use_qlora")
            if isinstance(flag, bool):
                return flag

    return _legacy_bnb_heuristic(adapter_dir)


def resolve_precision_safety(
    adapter_dir: Path,
    *,
    merged: bool,
    dequantize_confirmed: bool,
    strict_training_run: bool = False,
) -> PrecisionSafetyDecision:
    """Resolve the merged-export precision gate for one adapter."""
    was_qlora = was_trained_with_qlora(
        adapter_dir,
        strict_training_run=strict_training_run,
    )
    if not merged:
        return PrecisionSafetyDecision(
            was_qlora=was_qlora,
            safe=True,
            reason="Separate-adapter export does not merge base weights.",
            requires_dequantize=False,
        )
    if not was_qlora:
        return PrecisionSafetyDecision(
            was_qlora=False,
            safe=True,
            reason="Plain LoRA merge is safe without dequantization.",
            requires_dequantize=False,
        )
    if dequantize_confirmed:
        return PrecisionSafetyDecision(
            was_qlora=True,
            safe=True,
            reason="QLoRA merge confirmed with --dequantize; merging in fp16.",
            requires_dequantize=True,
        )
    return PrecisionSafetyDecision(
        was_qlora=True,
        safe=False,
        reason=_UNSAFE_MERGE_MESSAGE,
        requires_dequantize=True,
    )


def require_dequantize_or_refuse(plan: Any, adapter_dir: Path) -> PrecisionSafetyDecision:
    """Raise `UnsafeMergeError` when a merged QLoRA export lacks opt-in."""
    decision = resolve_precision_safety(
        adapter_dir,
        merged=bool(getattr(plan, "merged", False)),
        dequantize_confirmed=bool(getattr(plan, "dequantize_confirmed", False)),
        strict_training_run=True,
    )
    if not decision.safe:
        raise UnsafeMergeError(decision.reason)
    return decision


def _legacy_bnb_heuristic(adapter_dir: Path) -> bool:
    """Backward-compatible fallback for adapters without `training_run.json`."""
    pinned_path = adapter_dir / "pinned_versions.json"
    if not pinned_path.exists():
        return False
    try:
        pinned = json.loads(pinned_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    bnb = pinned.get("bitsandbytes")
    return isinstance(bnb, str) and bool(bnb)
