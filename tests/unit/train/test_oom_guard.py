"""OOM guard — recommendation math + message formatting."""

from __future__ import annotations

from dlm.train.oom_guard import format_oom_message, recommend_grad_accum


class TestRecommendGradAccum:
    def test_doubles_on_close_call(self) -> None:
        """Peak ≈ free → recommend at least 2×."""
        r = recommend_grad_accum(
            peak_bytes=22_000_000_000,
            free_at_start_bytes=22_000_000_000,
            current_grad_accum=2,
        )
        assert r >= 4

    def test_scales_with_ratio(self) -> None:
        """Peak = 4× free → recommend ~4× current."""
        r = recommend_grad_accum(
            peak_bytes=40_000_000_000,
            free_at_start_bytes=10_000_000_000,
            current_grad_accum=2,
        )
        assert r >= 8

    def test_zero_info_defaults_to_double(self) -> None:
        r = recommend_grad_accum(peak_bytes=0, free_at_start_bytes=0, current_grad_accum=4)
        assert r == 8

    def test_clamps_current_below_one(self) -> None:
        """Nonsense `current_grad_accum=0` is treated as 1."""
        r = recommend_grad_accum(peak_bytes=0, free_at_start_bytes=0, current_grad_accum=0)
        assert r == 2


class TestFormatOomMessage:
    def test_includes_step_and_numbers(self) -> None:
        msg = format_oom_message(
            step=12,
            peak_bytes=22_100_000_000,
            free_at_start_bytes=22_400_000_000,
            current_grad_accum=2,
            recommended_grad_accum=4,
        )
        assert "step 12" in msg
        assert "22.1 GB" in msg
        assert "22.4 GB" in msg
        assert "`grad_accum: 4`" in msg
        assert "was 2" in msg
        assert "--resume" in msg

    def test_multi_line(self) -> None:
        msg = format_oom_message(
            step=1,
            peak_bytes=1,
            free_at_start_bytes=1,
            current_grad_accum=1,
            recommended_grad_accum=2,
        )
        assert msg.count("\n") == 2
