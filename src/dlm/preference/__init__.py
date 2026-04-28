"""Preference-mining substrate types.

Pure-value surface only — typed contracts the mining/apply/train loop
builds on. The side-effecting mine/apply runtime lives in sibling
modules.
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
    JudgeInvocationError,
    JudgeUnavailableError,
    PreferenceMiningError,
)
from dlm.preference.judge import (
    CliJudge,
    HfRewardModelJudge,
    JudgeRef,
    PairScore,
    PreferenceJudge,
    SwayJudge,
    build_judge,
    parse_judge_ref,
)
from dlm.preference.mine import (
    PlannedMinedPreference,
    PreferenceMinePlan,
    PreferenceMinePrompt,
    PreferenceMineSkipReason,
    PreferenceMiningBackend,
    SkippedMinePrompt,
    build_mine_plan,
    render_mine_plan,
)

__all__ = [
    "InvalidJudgeSpecError",
    "JudgeInvocationError",
    "JudgeRef",
    "CliJudge",
    "HfRewardModelJudge",
    "PlannedPreferenceAddition",
    "PlannedMinedPreference",
    "PreferenceApplyPlan",
    "PreferenceApplySummary",
    "PreferenceMinePlan",
    "PreferenceMinePrompt",
    "PreferenceMiningBackend",
    "PreferenceMineSkipReason",
    "JudgeUnavailableError",
    "PairScore",
    "PreferenceJudge",
    "PreferenceMiningError",
    "PreferenceSkipReason",
    "SwayJudge",
    "SkippedPreferenceSection",
    "apply_plan",
    "build_apply_plan",
    "build_judge",
    "build_mine_plan",
    "parse_judge_ref",
    "render_apply_plan",
    "render_mine_plan",
    "revert_all_auto_mined",
    "SkippedMinePrompt",
]
