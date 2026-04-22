"""Runtime selection and SFTConfig-overrides for CPT refinements."""

from __future__ import annotations

import pytest

from dlm.train.cpt.runtime import (
    cpt_row_fraction,
    dapt_sft_config_overrides,
    row_mode,
    select_schedule,
)


class TestRowMode:
    def test_prose_row(self) -> None:
        assert row_mode({"text": "hello"}) == "cpt"

    def test_instruction_row(self) -> None:
        assert row_mode({"messages": [{"role": "user", "content": "hi"}]}) == "sft"

    def test_preference_row(self) -> None:
        assert row_mode({"prompt": "?", "chosen": "a", "rejected": "b"}) == "other"

    def test_text_none_is_not_cpt(self) -> None:
        assert row_mode({"text": None}) == "other"


class TestCptRowFraction:
    def test_all_cpt_is_one(self) -> None:
        rows = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        assert cpt_row_fraction(rows) == pytest.approx(1.0)

    def test_all_sft_is_zero(self) -> None:
        rows = [{"messages": []}, {"messages": []}]
        assert cpt_row_fraction(rows) == 0.0

    def test_mix(self) -> None:
        rows = [{"text": "a"}, {"text": "b"}, {"messages": []}, {"messages": []}]
        assert cpt_row_fraction(rows) == pytest.approx(0.5)

    def test_preference_ignored_in_denominator(self) -> None:
        # 1 cpt + 1 sft + 2 preference → 1/2 = 0.5 (preference not counted)
        rows = [
            {"text": "a"},
            {"messages": []},
            {"prompt": "?", "chosen": "c", "rejected": "r"},
            {"prompt": "?", "chosen": "c", "rejected": "r"},
        ]
        assert cpt_row_fraction(rows) == pytest.approx(0.5)

    def test_empty_iterable(self) -> None:
        assert cpt_row_fraction([]) == 0.0

    def test_only_preference_yields_zero(self) -> None:
        rows = [{"prompt": "?", "chosen": "c", "rejected": "r"}]
        assert cpt_row_fraction(rows) == 0.0


class TestSelectSchedule:
    @pytest.mark.parametrize("fraction", [0.0, 0.5, 0.71, 1.0])
    def test_dapt_explicit_wins_regardless_of_fraction(self, fraction: float) -> None:
        assert select_schedule("dapt", fraction) == "dapt"

    @pytest.mark.parametrize("fraction", [0.0, 0.71, 1.0])
    def test_sft_explicit_wins_regardless_of_fraction(self, fraction: float) -> None:
        assert select_schedule("sft", fraction) == "sft"

    def test_auto_below_threshold_picks_sft(self) -> None:
        assert select_schedule("auto", 0.69) == "sft"
        assert select_schedule("auto", 0.70) == "sft"  # strictly greater than

    def test_auto_above_threshold_picks_dapt(self) -> None:
        assert select_schedule("auto", 0.71) == "dapt"
        assert select_schedule("auto", 1.0) == "dapt"

    def test_unknown_setting_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown schedule setting"):
            select_schedule("warmup", 0.5)  # type: ignore[arg-type]


class TestDaptOverrides:
    def test_uses_cosine_with_min_lr(self) -> None:
        ov = dapt_sft_config_overrides()
        assert ov["lr_scheduler_type"] == "cosine_with_min_lr"

    def test_warmup_steps_default_20_percent_ratio(self) -> None:
        ov = dapt_sft_config_overrides()
        assert ov["warmup_steps"] == pytest.approx(0.2)

    def test_floor_ratio_default(self) -> None:
        ov = dapt_sft_config_overrides()
        assert ov["lr_scheduler_kwargs"] == {"min_lr_rate": pytest.approx(0.1)}

    def test_custom_floor_respected(self) -> None:
        ov = dapt_sft_config_overrides(floor_ratio=0.25)
        assert ov["lr_scheduler_kwargs"]["min_lr_rate"] == pytest.approx(0.25)

    def test_custom_warmup_respected(self) -> None:
        ov = dapt_sft_config_overrides(warmup_ratio=0.3)
        assert ov["warmup_steps"] == pytest.approx(0.3)
