"""Preference-mining substrate types.

Sprint 42 builds the mining/apply/train loop on top of these typed
contracts. This module only exposes the pure-value surface; the
side-effecting mine/apply runtime lands in follow-up slices.
"""

from dlm.preference.apply import (
    PlannedPreferenceAddition,
    PreferenceApplyPlan,
    PreferenceApplySummary,
    PreferenceSkipReason,
    SkippedPreferenceSection,
    apply_plan,
    build_apply_plan,
    render_apply_plan,
    revert_all_auto_mined,
)
from dlm.preference.errors import (
    InvalidJudgeSpecError,
    JudgeUnavailableError,
    PreferenceMiningError,
)
from dlm.preference.judge import JudgeRef, PairScore, PreferenceJudge, parse_judge_ref

__all__ = [
    "InvalidJudgeSpecError",
    "JudgeRef",
    "PlannedPreferenceAddition",
    "PreferenceApplyPlan",
    "PreferenceApplySummary",
    "JudgeUnavailableError",
    "PairScore",
    "PreferenceJudge",
    "PreferenceMiningError",
    "PreferenceSkipReason",
    "SkippedPreferenceSection",
    "apply_plan",
    "build_apply_plan",
    "parse_judge_ref",
    "render_apply_plan",
    "revert_all_auto_mined",
]
