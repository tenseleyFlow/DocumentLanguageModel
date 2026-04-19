"""ExportPlan — quant validation, QLoRA-merge safety gate."""

from __future__ import annotations

import pytest

from dlm.export.errors import UnsafeMergeError
from dlm.export.plan import (
    DEFAULT_QUANT,
    QUANT_BYTES_PER_PARAM,
    ExportPlan,
    resolve_export_plan,
    valid_quants,
)


class TestExportPlanValidation:
    def test_default_quant_is_q4km(self) -> None:
        plan = ExportPlan()
        assert plan.quant == "Q4_K_M"
        assert plan.quant == DEFAULT_QUANT

    def test_unknown_quant_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown quant"):
            ExportPlan(quant="Q3_K_S")  # type: ignore[arg-type]

    def test_all_documented_quants_valid(self) -> None:
        for q in valid_quants():
            ExportPlan(quant=q)  # type: ignore[arg-type]

    def test_dequantize_without_merged_rejected(self) -> None:
        with pytest.raises(ValueError, match="--dequantize only makes sense"):
            ExportPlan(dequantize_confirmed=True, merged=False)

    def test_dequantize_with_merged_ok(self) -> None:
        plan = ExportPlan(merged=True, dequantize_confirmed=True)
        assert plan.merged is True
        assert plan.dequantize_confirmed is True


class TestMergeSafetyGate:
    def test_plain_lora_merge_never_raises(self) -> None:
        plan = ExportPlan(merged=True)
        plan.assert_merge_safe(was_qlora=False)

    def test_qlora_merge_without_dequantize_raises(self) -> None:
        plan = ExportPlan(merged=True, dequantize_confirmed=False)
        with pytest.raises(UnsafeMergeError, match="QLoRA"):
            plan.assert_merge_safe(was_qlora=True)

    def test_qlora_merge_with_dequantize_ok(self) -> None:
        plan = ExportPlan(merged=True, dequantize_confirmed=True)
        plan.assert_merge_safe(was_qlora=True)

    def test_non_merged_plan_never_raises(self) -> None:
        plan = ExportPlan(merged=False)
        plan.assert_merge_safe(was_qlora=True)
        plan.assert_merge_safe(was_qlora=False)


class TestEstimatedBytes:
    def test_q4km_ratio_matches_table(self) -> None:
        plan = ExportPlan(quant="Q4_K_M")
        # 1.5B params at 0.56 bytes/param ≈ 840MB.
        assert plan.estimated_base_bytes(1_500_000_000) == int(
            1_500_000_000 * QUANT_BYTES_PER_PARAM["Q4_K_M"]
        )


class TestResolveExportPlan:
    def test_cli_beats_frontmatter_beats_default(self) -> None:
        plan = resolve_export_plan(
            cli_quant="Q5_K_M",
            cli_merged=False,
            cli_dequantize=False,
            cli_no_template=False,
            cli_ollama_name=None,
            frontmatter_default_quant="Q6_K",
        )
        assert plan.quant == "Q5_K_M"

    def test_frontmatter_wins_over_default_when_cli_none(self) -> None:
        plan = resolve_export_plan(
            cli_quant=None,
            cli_merged=False,
            cli_dequantize=False,
            cli_no_template=False,
            cli_ollama_name=None,
            frontmatter_default_quant="Q6_K",
        )
        assert plan.quant == "Q6_K"

    def test_fallback_to_default_when_everything_none(self) -> None:
        plan = resolve_export_plan(
            cli_quant=None,
            cli_merged=False,
            cli_dequantize=False,
            cli_no_template=False,
            cli_ollama_name=None,
            frontmatter_default_quant=None,
        )
        assert plan.quant == DEFAULT_QUANT

    def test_invalid_quant_rejected_at_resolve_time(self) -> None:
        with pytest.raises(ValueError, match="unknown quant"):
            resolve_export_plan(
                cli_quant="Q3_XS",
                cli_merged=False,
                cli_dequantize=False,
                cli_no_template=False,
                cli_ollama_name=None,
                frontmatter_default_quant=None,
            )

    def test_no_template_flag_inverts_include_template(self) -> None:
        plan = resolve_export_plan(
            cli_quant=None,
            cli_merged=False,
            cli_dequantize=False,
            cli_no_template=True,
            cli_ollama_name=None,
            frontmatter_default_quant=None,
        )
        assert plan.include_template is False
