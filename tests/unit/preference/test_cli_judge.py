"""Unit tests for Sprint 42's external CLI judge runtime."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from dlm.preference import (
    CliJudge,
    HfRewardModelJudge,
    JudgeInvocationError,
    JudgeUnavailableError,
    SwayJudge,
    build_judge,
)


def _proc(
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["judge-bin"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class TestCliJudge:
    def test_scores_pair_via_two_json_round_trips(self) -> None:
        seen_payloads: list[str] = []

        def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            argv = args[0]
            assert argv == ["judge-bin", "--json"]
            payload = kwargs["input"]
            assert isinstance(payload, str)
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _proc(stdout='{"score": 0.2, "reasoning": "too vague"}')
            return _proc(stdout='{"score": 0.9, "reasoning": "specific and correct"}')

        judge = CliJudge("judge-bin --json")
        with patch("dlm.preference.judge.subprocess.run", side_effect=fake_run):
            score = judge.score_pair("What is DGEMM?", "bad", "good")

        assert score.score_a == pytest.approx(0.2)
        assert score.score_b == pytest.approx(0.9)
        assert score.preferred == "b"
        assert "a: too vague" in (score.reasoning or "")
        assert "b: specific and correct" in (score.reasoning or "")
        assert '"prompt": "What is DGEMM?"' in seen_payloads[0]
        assert '"candidate": "bad"' in seen_payloads[0]
        assert '"candidate": "good"' in seen_payloads[1]

    def test_non_zero_exit_raises(self) -> None:
        judge = CliJudge("judge-bin")
        with (
            patch(
                "dlm.preference.judge.subprocess.run",
                return_value=_proc(returncode=7, stderr="bad model"),
            ),
            pytest.raises(JudgeInvocationError, match="exited 7: bad model"),
        ):
            judge.score_pair("p", "a", "b")

    def test_invalid_json_raises(self) -> None:
        judge = CliJudge("judge-bin")
        with (
            patch(
                "dlm.preference.judge.subprocess.run",
                return_value=_proc(stdout="not-json"),
            ),
            pytest.raises(JudgeInvocationError, match="invalid JSON"),
        ):
            judge.score_pair("p", "a", "b")

    def test_missing_numeric_score_raises(self) -> None:
        judge = CliJudge("judge-bin")
        with (
            patch(
                "dlm.preference.judge.subprocess.run",
                return_value=_proc(stdout='{"reasoning": "oops"}'),
            ),
            pytest.raises(JudgeInvocationError, match="numeric `score`"),
        ):
            judge.score_pair("p", "a", "b")

    def test_missing_binary_raises_unavailable(self) -> None:
        judge = CliJudge("judge-bin")
        with (
            patch(
                "dlm.preference.judge.subprocess.run",
                side_effect=FileNotFoundError("judge-bin"),
            ),
            pytest.raises(JudgeUnavailableError, match="not available on PATH"),
        ):
            judge.score_pair("p", "a", "b")

    def test_timeout_raises_invocation_error(self) -> None:
        judge = CliJudge("judge-bin", timeout=1.5)
        with (
            patch(
                "dlm.preference.judge.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="judge-bin", timeout=1.5),
            ),
            pytest.raises(JudgeInvocationError, match="timed out after 1.5s"),
        ):
            judge.score_pair("p", "a", "b")


class TestBuildJudge:
    def test_cli_ref_builds_concrete_cli_judge(self) -> None:
        judge = build_judge("cli:judge-bin --json")

        assert isinstance(judge, CliJudge)
        assert judge.name == "cli:judge-bin --json"

    def test_hf_ref_builds_concrete_hf_judge(self) -> None:
        judge = build_judge("hf:reward/model")

        assert isinstance(judge, HfRewardModelJudge)
        assert judge.name == "hf:reward/model"

    def test_sway_ref_builds_concrete_sway_judge(self) -> None:
        judge = build_judge("sway", dlm_path=Path("/tmp/example.dlm"))

        assert isinstance(judge, SwayJudge)
        assert judge.name == "sway:preference_judge"

    def test_sway_ref_requires_dlm_path_context(self) -> None:
        with pytest.raises(JudgeUnavailableError, match="requires the .dlm path context"):
            build_judge("sway")
