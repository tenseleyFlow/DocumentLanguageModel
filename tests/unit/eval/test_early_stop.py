"""EarlyStopConfig validation + was_early_stopped heuristic."""

from __future__ import annotations

import pytest

from dlm.eval.early_stop import EarlyStopConfig, build_callback, was_early_stopped


class TestEarlyStopConfig:
    def test_defaults(self) -> None:
        cfg = EarlyStopConfig()
        assert cfg.patience == 3
        assert cfg.threshold == 0.0
        assert cfg.metric == "eval_loss"
        assert cfg.greater_is_better is False

    def test_patience_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="patience"):
            EarlyStopConfig(patience=0)

    def test_negative_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            EarlyStopConfig(threshold=-0.1)

    def test_empty_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            EarlyStopConfig(metric="")


class TestBuildCallback:
    def test_returns_hf_callback(self) -> None:
        cfg = EarlyStopConfig(patience=5, threshold=0.01)
        callback = build_callback(cfg)
        # HF's EarlyStoppingCallback stores these as attributes.
        assert callback.early_stopping_patience == 5
        assert callback.early_stopping_threshold == pytest.approx(0.01)


class TestWasEarlyStopped:
    def test_max_steps_hit_exactly_means_not_stopped(self) -> None:
        assert not was_early_stopped(
            max_steps_ran=100, configured_max_steps=100, num_epochs_done=0.4
        )

    def test_max_steps_not_hit_means_stopped(self) -> None:
        assert was_early_stopped(max_steps_ran=47, configured_max_steps=100, num_epochs_done=0.2)

    def test_integer_epochs_mean_not_stopped(self) -> None:
        """Natural completion finishes exactly `num_train_epochs`."""
        assert not was_early_stopped(
            max_steps_ran=500, configured_max_steps=None, num_epochs_done=3.0
        )

    def test_fractional_epoch_means_stopped(self) -> None:
        assert was_early_stopped(max_steps_ran=200, configured_max_steps=None, num_epochs_done=1.47)

    def test_max_steps_zero_falls_back_to_epoch_check(self) -> None:
        """`max_steps=0` (or negative) means the cap isn't active."""
        assert not was_early_stopped(max_steps_ran=300, configured_max_steps=0, num_epochs_done=2.0)
