"""Perplexity utility — finite, inf, nan, negative, overflow guards."""

from __future__ import annotations

import math

import pytest

from dlm.eval.perplexity import perplexity


class TestPerplexity:
    def test_zero_loss_gives_one(self) -> None:
        assert perplexity(0.0) == pytest.approx(1.0)

    def test_positive_loss_gives_exp(self) -> None:
        assert perplexity(1.0) == pytest.approx(math.e)
        assert perplexity(math.log(10.0)) == pytest.approx(10.0)

    def test_nan_returns_inf(self) -> None:
        assert perplexity(math.nan) == math.inf

    def test_inf_returns_inf(self) -> None:
        assert perplexity(math.inf) == math.inf

    def test_negative_returns_inf(self) -> None:
        """Negative cross-entropy loss is nonsense — report inf, not a tiny PPL."""
        assert perplexity(-1.0) == math.inf

    def test_overflow_returns_inf(self) -> None:
        """`exp(1000.0)` overflows; we substitute inf."""
        assert perplexity(1000.0) == math.inf
