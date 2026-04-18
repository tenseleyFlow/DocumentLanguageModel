"""Hardware capabilities + training-plan resolution.

Public surface: `doctor()` for CLI use, `probe()` for just capabilities,
`resolve()` for producing a `TrainingPlan` from a config + caps, and
`ResolutionError` for refusal handling.

Backend detection lives in `dlm.hardware.backend` and stays minimal so
cold CLI paths (`dlm --version`) don't pay for the full probe.
"""

from __future__ import annotations

from dlm.hardware.backend import Backend, detect
from dlm.hardware.capabilities import Capabilities, DeterminismClass, probe
from dlm.hardware.doctor import (
    DEFAULT_REFERENCE_BASE_PARAMS,
    DEFAULT_REFERENCE_SEQ_LEN,
    DoctorResult,
    doctor,
)
from dlm.hardware.memory import estimate_peak_vram_gb, estimate_step_seconds
from dlm.hardware.plan import TrainingPlan, resolve
from dlm.hardware.refusals import CPU_PARAM_BUDGET, ResolutionError, check_refusals
from dlm.hardware.render import render_text

__all__ = [
    "CPU_PARAM_BUDGET",
    "Backend",
    "Capabilities",
    "DEFAULT_REFERENCE_BASE_PARAMS",
    "DEFAULT_REFERENCE_SEQ_LEN",
    "DeterminismClass",
    "DoctorResult",
    "ResolutionError",
    "TrainingPlan",
    "check_refusals",
    "detect",
    "doctor",
    "estimate_peak_vram_gb",
    "estimate_step_seconds",
    "probe",
    "render_text",
    "resolve",
]
