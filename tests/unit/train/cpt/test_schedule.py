"""Unit tests for the cosine-with-floor LR schedule."""

from __future__ import annotations

import math

import pytest

from dlm.train.cpt.schedule import (
    DAPT_FLOOR_RATIO,
    DAPT_WARMUP_RATIO,
    cosine_with_floor_lr,
    dapt_warmup_steps,
)


class TestWarmupRamp:
    def test_step_zero_is_zero(self) -> None:
        assert cosine_with_floor_lr(0, total_steps=100, warmup_steps=20) == 0.0

    def test_half_warmup_is_half(self) -> None:
        assert cosine_with_floor_lr(
            10, total_steps=100, warmup_steps=20
        ) == pytest.approx(0.5)

    def test_warmup_end_is_peak(self) -> None:
        # step == warmup_steps: first step of decay phase; cosine is 1.0
        # at decay_progress=0, so we're at peak.
        assert cosine_with_floor_lr(
            20, total_steps=100, warmup_steps=20
        ) == pytest.approx(1.0)

    def test_zero_warmup_jumps_to_peak(self) -> None:
        assert cosine_with_floor_lr(
            0, total_steps=100, warmup_steps=0
        ) == pytest.approx(1.0)


class TestCosineDecay:
    def test_monotone_decrease_through_decay(self) -> None:
        lrs = [
            cosine_with_floor_lr(s, total_steps=100, warmup_steps=20)
            for s in range(20, 100, 5)
        ]
        for a, b in zip(lrs, lrs[1:], strict=False):
            assert a > b

    def test_end_step_hits_floor(self) -> None:
        # step == total_steps: the clamp kicks in at floor.
        assert cosine_with_floor_lr(
            100, total_steps=100, warmup_steps=20, floor_ratio=0.1
        ) == pytest.approx(0.1)

    def test_beyond_end_clamped_at_floor(self) -> None:
        assert cosine_with_floor_lr(
            500, total_steps=100, warmup_steps=20, floor_ratio=0.25
        ) == pytest.approx(0.25)

    def test_floor_ratio_zero_reaches_zero_at_end(self) -> None:
        # With floor_ratio=0 the schedule matches a vanilla cosine.
        assert cosine_with_floor_lr(
            100, total_steps=100, warmup_steps=20, floor_ratio=0.0
        ) == pytest.approx(0.0)

    def test_midpoint_is_halfway_between_peak_and_floor(self) -> None:
        # Midpoint of cosine decay (decay_progress=0.5) gives cos(pi/2)=0,
        # so cosine multiplier = 0.5 → LR = floor + (1-floor)*0.5
        floor = 0.1
        mid = cosine_with_floor_lr(
            60, total_steps=100, warmup_steps=20, floor_ratio=floor
        )
        expected = floor + (1.0 - floor) * 0.5
        assert mid == pytest.approx(expected)


class TestInputValidation:
    def test_negative_step_rejected(self) -> None:
        with pytest.raises(ValueError, match="step must be non-negative"):
            cosine_with_floor_lr(-1, total_steps=100, warmup_steps=10)

    def test_zero_total_steps_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_steps must be positive"):
            cosine_with_floor_lr(0, total_steps=0, warmup_steps=0)

    def test_warmup_equal_to_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be < total_steps"):
            cosine_with_floor_lr(0, total_steps=100, warmup_steps=100)

    def test_warmup_greater_than_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be < total_steps"):
            cosine_with_floor_lr(0, total_steps=100, warmup_steps=200)

    def test_negative_warmup_rejected(self) -> None:
        with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
            cosine_with_floor_lr(0, total_steps=100, warmup_steps=-1)

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
    def test_floor_ratio_out_of_range(self, bad: float) -> None:
        with pytest.raises(ValueError, match="floor_ratio must be in"):
            cosine_with_floor_lr(
                0, total_steps=100, warmup_steps=10, floor_ratio=bad
            )


class TestDefaultConstants:
    def test_default_floor_matches_literature(self) -> None:
        assert DAPT_FLOOR_RATIO == pytest.approx(0.1)

    def test_default_warmup_ratio(self) -> None:
        assert DAPT_WARMUP_RATIO == pytest.approx(0.2)


class TestDaptWarmupSteps:
    def test_default_ratio_yields_20_percent(self) -> None:
        assert dapt_warmup_steps(100) == 20

    def test_small_total_rounds_to_at_least_one(self) -> None:
        assert dapt_warmup_steps(3) == 1  # 3 * 0.2 = 0.6 → 1

    def test_zero_total_gives_zero(self) -> None:
        assert dapt_warmup_steps(0) == 0

    def test_custom_ratio_respected(self) -> None:
        assert dapt_warmup_steps(50, ratio=0.5) == 25


class TestContinuityAcrossWarmup:
    def test_last_warmup_step_and_first_decay_step_are_close(self) -> None:
        # Just before warmup ends, ramp says step/warmup ≈ nearly 1.
        # At warmup_steps, cosine is exactly 1. They should differ by
        # ~1/warmup_steps (the ramp's last sub-peak increment).
        warmup = 50
        last_ramp = cosine_with_floor_lr(
            warmup - 1, total_steps=200, warmup_steps=warmup
        )
        first_decay = cosine_with_floor_lr(
            warmup, total_steps=200, warmup_steps=warmup
        )
        assert first_decay == pytest.approx(1.0)
        assert math.isclose(first_decay - last_ramp, 1 / warmup, abs_tol=1e-9)
