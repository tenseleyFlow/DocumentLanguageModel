"""Per-rank I/O helpers — `master_only`, `barrier`, `gather_metrics`."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dlm.train.distributed.rank_io import (
    barrier,
    gather_metrics,
    is_main_process,
    master_only,
)


class TestIsMainProcess:
    def test_none_accelerator_returns_true(self) -> None:
        assert is_main_process(None) is True

    def test_rank_0_returns_true(self) -> None:
        assert is_main_process(SimpleNamespace(is_main_process=True)) is True

    def test_non_zero_rank_returns_false(self) -> None:
        assert is_main_process(SimpleNamespace(is_main_process=False)) is False

    def test_missing_attribute_defaults_to_true(self) -> None:
        """Missing `is_main_process` attr → assume single-process → True."""
        assert is_main_process(SimpleNamespace()) is True


class TestMasterOnly:
    def test_rank_0_calls_function(self) -> None:
        acc = SimpleNamespace(is_main_process=True)
        calls: list[int] = []

        @master_only
        def write(_acc: object, value: int) -> int:
            calls.append(value)
            return value

        result = write(acc, 42)
        assert result == 42
        assert calls == [42]

    def test_non_zero_rank_skips(self) -> None:
        acc = SimpleNamespace(is_main_process=False)
        calls: list[int] = []

        @master_only
        def write(_acc: object, value: int) -> int:
            calls.append(value)
            return value

        result = write(acc, 42)
        assert result is None
        assert calls == []

    def test_single_process_calls_function(self) -> None:
        calls: list[int] = []

        @master_only
        def write(_acc: object, value: int) -> int:
            calls.append(value)
            return value

        assert write(None, 7) == 7
        assert calls == [7]


class TestBarrier:
    def test_none_accelerator_is_noop(self) -> None:
        # Should not raise, does nothing.
        barrier(None)

    def test_calls_wait_for_everyone(self) -> None:
        acc = MagicMock()
        acc.wait_for_everyone = MagicMock()
        barrier(acc)
        acc.wait_for_everyone.assert_called_once()

    def test_tolerates_missing_method(self) -> None:
        # Stub accelerator with no `wait_for_everyone` — old versions
        # or mocks. Must not raise.
        barrier(SimpleNamespace())


class TestGatherMetrics:
    def test_none_accelerator_passes_through(self) -> None:
        out = gather_metrics(None, {"loss": 2.5, "ppl": 12.0})
        assert out == {"loss": 2.5, "ppl": 12.0}

    def test_missing_gather_method_passes_through(self) -> None:
        """Accelerator without `gather_for_metrics` degrades gracefully."""
        out = gather_metrics(SimpleNamespace(), {"loss": 1.0})
        assert out == {"loss": 1.0}

    def test_gather_averaged_across_ranks(self) -> None:
        """Simulate 2 ranks — gather returns a stacked tensor; take mean."""
        import torch

        def _gather(tensor: torch.Tensor) -> torch.Tensor:
            # Simulate two-rank gather: stack the input with a twin.
            return torch.stack([tensor, tensor + 1.0])

        acc = SimpleNamespace(gather_for_metrics=_gather, is_main_process=True)
        out = gather_metrics(acc, {"loss": 2.0})
        # mean of [2.0, 3.0] = 2.5
        assert out["loss"] == pytest.approx(2.5)

    def test_gather_none_falls_back_to_original_value(self) -> None:
        acc = SimpleNamespace(gather_for_metrics=lambda tensor: None, is_main_process=True)
        out = gather_metrics(acc, {"loss": 2.0})
        assert out == {"loss": 2.0}
