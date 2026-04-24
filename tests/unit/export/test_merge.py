"""Pure merge helper coverage."""

from __future__ import annotations

import pytest

from dlm.export.errors import UnsafeMergeError
from dlm.export.merge import check_merge_safety
from dlm.export.plan import ExportPlan


def test_check_merge_safety_delegates_to_plan() -> None:
    check_merge_safety(ExportPlan(merged=False), was_qlora=True)


def test_check_merge_safety_refuses_unsafe_qlora_merge() -> None:
    with pytest.raises(UnsafeMergeError, match="QLoRA"):
        check_merge_safety(
            ExportPlan(merged=True, dequantize_confirmed=False),
            was_qlora=True,
        )
