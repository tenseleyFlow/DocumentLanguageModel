"""Pure-value substrate tests for Sprint 42 judge types."""

from __future__ import annotations

import pytest

from dlm.preference import InvalidJudgeSpecError, PairScore, parse_judge_ref


class TestPairScore:
    def test_margin_is_computed(self) -> None:
        score = PairScore(score_a=0.75, score_b=0.25, reasoning="clear winner")
        assert score.margin == pytest.approx(0.5)
        assert score.preferred == "a"
        assert score.reasoning == "clear winner"

    def test_tie_reports_tie(self) -> None:
        score = PairScore(score_a=0.4, score_b=0.4)
        assert score.margin == pytest.approx(0.0)
        assert score.preferred == "tie"

    @pytest.mark.parametrize(("a", "b"), [(float("nan"), 0.0), (0.0, float("inf"))])
    def test_non_finite_scores_refused(self, a: float, b: float) -> None:
        with pytest.raises(ValueError, match="finite floats"):
            PairScore(score_a=a, score_b=b)


class TestParseJudgeRef:
    def test_parses_sway(self) -> None:
        ref = parse_judge_ref("sway")
        assert ref.kind == "sway"
        assert ref.target is None

    def test_parses_hf(self) -> None:
        ref = parse_judge_ref("hf:OpenAssistant/reward-model")
        assert ref.kind == "hf"
        assert ref.target == "OpenAssistant/reward-model"

    def test_parses_cli(self) -> None:
        ref = parse_judge_ref("cli:/usr/local/bin/judge --json")
        assert ref.kind == "cli"
        assert ref.target == "/usr/local/bin/judge --json"

    @pytest.mark.parametrize("raw", ["", "hf:", "cli:", "bogus"])
    def test_bad_specs_refused(self, raw: str) -> None:
        with pytest.raises(InvalidJudgeSpecError):
            parse_judge_ref(raw)
