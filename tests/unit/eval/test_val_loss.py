"""compute_metrics hook + summarize_eval_state."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from dlm.eval.val_loss import eval_metrics_from_eval_pred, summarize_eval_state


class TestEvalMetricsFromEvalPred:
    def test_returns_perplexity(self) -> None:
        pred = SimpleNamespace(metrics={"eval_loss": math.log(10.0)})
        result = eval_metrics_from_eval_pred(pred)
        assert result["perplexity"] == pytest.approx(10.0)

    def test_missing_metrics_returns_empty(self) -> None:
        pred = SimpleNamespace(metrics=None)
        assert eval_metrics_from_eval_pred(pred) == {}

    def test_non_numeric_loss_returns_empty(self) -> None:
        pred = SimpleNamespace(metrics={"eval_loss": "not a number"})
        assert eval_metrics_from_eval_pred(pred) == {}

    def test_no_metrics_attr(self) -> None:
        """An EvalPrediction without `metrics` attribute returns empty."""
        pred = SimpleNamespace()
        assert eval_metrics_from_eval_pred(pred) == {}


class TestSummarizeEvalState:
    def test_picks_last_eval_loss(self) -> None:
        history = [
            {"loss": 2.5, "step": 10},
            {"eval_loss": 2.1, "step": 10},
            {"loss": 2.3, "step": 20},
            {"eval_loss": 1.9, "step": 20},
        ]
        result = summarize_eval_state(history)
        assert result["final_val_loss"] == pytest.approx(1.9)
        assert result["final_val_perplexity"] == pytest.approx(math.exp(1.9))

    def test_no_eval_loss_in_history(self) -> None:
        result = summarize_eval_state([{"loss": 1.0}])
        assert result["final_val_loss"] is None
        assert result["final_val_perplexity"] is None

    def test_empty_history(self) -> None:
        result = summarize_eval_state([])
        assert result["final_val_loss"] is None
        assert result["final_val_perplexity"] is None

    def test_non_numeric_eval_loss_skipped(self) -> None:
        history = [
            {"eval_loss": "oops", "step": 10},
            {"eval_loss": 1.5, "step": 20},
        ]
        result = summarize_eval_state(history)
        assert result["final_val_loss"] == pytest.approx(1.5)
