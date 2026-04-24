"""Preference-mining substrate types.

Sprint 42 builds the mining/apply/train loop on top of these typed
contracts. This module only exposes the pure-value surface; the
side-effecting mine/apply runtime lands in follow-up slices.
"""

from dlm.preference.errors import (
    InvalidJudgeSpecError,
    JudgeUnavailableError,
    PreferenceMiningError,
)
from dlm.preference.judge import JudgeRef, PairScore, PreferenceJudge, parse_judge_ref

__all__ = [
    "InvalidJudgeSpecError",
    "JudgeRef",
    "JudgeUnavailableError",
    "PairScore",
    "PreferenceJudge",
    "PreferenceMiningError",
    "parse_judge_ref",
]
