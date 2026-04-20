"""`summarize_eval_state` must drop NaN/inf eval_loss entries.

Previously, the function passed NaN through as a valid `float`, so the
TrainingSummary carried `final_val_loss: NaN` (serialized as the
non-JSON-standard token `NaN` on disk) and no gate fired. Now NaN
entries are skipped and the trainer orchestrator asserts separately
via `assert_eval_finite` before the commit step.
"""

from __future__ import annotations

import math

from dlm.eval.val_loss import summarize_eval_state


def test_finite_eval_is_preserved() -> None:
    out = summarize_eval_state([{"eval_loss": 1.8, "step": 10}])
    assert out["final_val_loss"] == 1.8
    assert out["final_val_perplexity"] is not None


def test_nan_eval_dropped_returns_none() -> None:
    out = summarize_eval_state([{"eval_loss": float("nan"), "step": 10}])
    assert out["final_val_loss"] is None
    assert out["final_val_perplexity"] is None


def test_inf_eval_dropped_returns_none() -> None:
    out = summarize_eval_state([{"eval_loss": float("inf"), "step": 10}])
    assert out["final_val_loss"] is None
    assert out["final_val_perplexity"] is None


def test_nan_final_falls_back_to_earlier_finite() -> None:
    # Walks back — skipping the NaN and using the finite entry.
    out = summarize_eval_state(
        [
            {"eval_loss": 1.5, "step": 10},
            {"eval_loss": float("nan"), "step": 20},
        ]
    )
    assert out["final_val_loss"] == 1.5


def test_no_eval_entries_returns_none() -> None:
    out = summarize_eval_state([{"loss": 2.0, "step": 1}])
    assert out["final_val_loss"] is None
    assert out["final_val_perplexity"] is None


def test_empty_log_returns_none() -> None:
    out = summarize_eval_state([])
    assert out["final_val_loss"] is None


def test_final_val_loss_is_never_nan_on_return() -> None:
    # Defensive: whatever we return must be JSON-safe (None or finite).
    out = summarize_eval_state(
        [
            {"eval_loss": float("nan"), "step": 10},
            {"eval_loss": float("inf"), "step": 20},
        ]
    )
    loss = out["final_val_loss"]
    assert loss is None or math.isfinite(loss)
