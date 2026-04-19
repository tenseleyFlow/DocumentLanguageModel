"""ExportPlan — quant validation, QLoRA-merge safety gate, imatrix mode."""

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


class TestImatrixMode:
    """Sprint 11.6 — imatrix field + needs_imatrix() predicate."""

    def test_default_is_auto(self) -> None:
        assert ExportPlan().imatrix == "auto"

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown imatrix mode"):
            ExportPlan(imatrix="fast")  # type: ignore[arg-type]

    @pytest.mark.parametrize("mode", ["auto", "off", "cached"])
    def test_all_modes_valid(self, mode: str) -> None:
        ExportPlan(imatrix=mode)  # type: ignore[arg-type]

    @pytest.mark.parametrize("quant", ["Q4_K_M", "Q5_K_M", "Q6_K"])
    def test_needs_imatrix_true_for_k_quants(self, quant: str) -> None:
        plan = ExportPlan(quant=quant)  # type: ignore[arg-type]
        assert plan.needs_imatrix() is True

    @pytest.mark.parametrize("quant", ["Q8_0", "F16"])
    def test_needs_imatrix_false_for_non_k(self, quant: str) -> None:
        """Upstream ignores --imatrix on non-k-quants; don't bother building."""
        plan = ExportPlan(quant=quant)  # type: ignore[arg-type]
        assert plan.needs_imatrix() is False

    def test_off_mode_suppresses(self) -> None:
        plan = ExportPlan(quant="Q4_K_M", imatrix="off")
        assert plan.needs_imatrix() is False

    def test_cached_mode_still_applicable(self) -> None:
        """`cached` doesn't skip; it just changes *how* we acquire the imatrix."""
        plan = ExportPlan(quant="Q4_K_M", imatrix="cached")
        assert plan.needs_imatrix() is True


class TestResolveImatrixFlag:
    def test_default_is_auto(self) -> None:
        plan = resolve_export_plan(
            cli_quant=None,
            cli_merged=False,
            cli_dequantize=False,
            cli_no_template=False,
            cli_ollama_name=None,
            frontmatter_default_quant=None,
        )
        assert plan.imatrix == "auto"

    def test_no_imatrix_flips_to_off(self) -> None:
        plan = resolve_export_plan(
            cli_quant=None,
            cli_merged=False,
            cli_dequantize=False,
            cli_no_template=False,
            cli_ollama_name=None,
            frontmatter_default_quant=None,
            cli_no_imatrix=True,
        )
        assert plan.imatrix == "off"


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
