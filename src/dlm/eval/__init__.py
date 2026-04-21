"""Eval harness — val loss, perplexity, probes, retention, early stopping.

See Sprint 10 for the design. Heavy imports are deferred to the
boundaries that need them.
"""

from __future__ import annotations

from dlm.eval.early_stop import EarlyStopConfig, build_callback, was_early_stopped
from dlm.eval.errors import EvalError, ProbeFormatError, RetentionSliceError
from dlm.eval.perplexity import perplexity
from dlm.eval.probes import Probe, extract_probes
from dlm.eval.retention import RetentionSlice, build_retention_slice, retention_delta
from dlm.eval.summary import (
    ProbeOutput,
    SourceProvenanceRecord,
    TrainingSummary,
    load_summary,
    save_summary,
    summary_path_for,
)
from dlm.eval.val_loss import eval_metrics_from_eval_pred, summarize_eval_state

__all__ = [
    "EarlyStopConfig",
    "EvalError",
    "Probe",
    "ProbeFormatError",
    "ProbeOutput",
    "RetentionSlice",
    "RetentionSliceError",
    "SourceProvenanceRecord",
    "TrainingSummary",
    "build_callback",
    "build_retention_slice",
    "eval_metrics_from_eval_pred",
    "extract_probes",
    "load_summary",
    "perplexity",
    "retention_delta",
    "save_summary",
    "summarize_eval_state",
    "summary_path_for",
    "was_early_stopped",
]
