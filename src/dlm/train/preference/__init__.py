"""Preference training (DPO).

The SFT `trainer.run()` owns supervised fine-tuning end-to-end. This
package owns the DPO phase: dataset assembly from `::preference::`
sections, DPOTrainer wiring, reference-adapter loading, and the
phase orchestrator that sequences SFT → DPO within a single
`dlm train` invocation.

Public re-exports are kept minimal — callers reach through named
modules (`dpo_dataset`, `dpo_trainer`, `phase_orchestrator`) so import
graphs stay explicit. Only the errors are re-exported.
"""

from __future__ import annotations

from dlm.train.preference.errors import (
    DpoPhaseError,
    DpoReferenceLoadError,
    NoPreferenceContentError,
    PriorAdapterRequiredError,
)

__all__ = [
    "DpoPhaseError",
    "DpoReferenceLoadError",
    "NoPreferenceContentError",
    "PriorAdapterRequiredError",
]
