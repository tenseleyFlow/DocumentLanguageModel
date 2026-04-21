"""GaLore optimizer enum acceptance + plan-reason warning."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlm.doc.schema import TrainingConfig


class TestSchemaAcceptsGalore:
    def test_accepts_galore_adamw(self) -> None:
        tc = TrainingConfig(optimizer="galore_adamw")
        assert tc.optimizer == "galore_adamw"

    def test_accepts_galore_adamw_8bit(self) -> None:
        tc = TrainingConfig(optimizer="galore_adamw_8bit")
        assert tc.optimizer == "galore_adamw_8bit"

    def test_classic_optimizers_still_accepted(self) -> None:
        for opt in ("adamw_torch", "adamw_bnb_8bit", "paged_adamw_8bit"):
            assert TrainingConfig(optimizer=opt).optimizer == opt

    def test_unknown_optimizer_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(optimizer="adam")  # type: ignore[arg-type]


class TestPlanReasonGaloreWarning:
    def _make_caps(self):  # type: ignore[no-untyped-def]
        from dlm.hardware import Capabilities
        from dlm.hardware.backend import Backend

        return Capabilities(
            backend=Backend.CUDA,
            device_name="test",
            sm=(8, 0),
            rocm_arch=None,
            vram_gb=40.0,
            unified_memory_gb=None,
            cpu_cores=8,
            ram_gb=64.0,
            supports_bf16=True,
            supports_fp16=True,
            has_flash_attention=False,
            has_xformers=False,
            has_bitsandbytes=False,
            has_triton=False,
            has_mlx=False,
            torch_version="2.5.0",
            accelerate_version="1.0.0",
            cuda_version="12.1",
            rocm_version=None,
            platform="linux",
            determinism_class="strong",
        )

    def test_small_base_galore_warns(self) -> None:
        from dlm.hardware.plan import resolve

        tc = TrainingConfig(optimizer="galore_adamw")
        plan = resolve(tc, self._make_caps(), seq_len=2048, base_params=135_000_000)
        assert "warn=galore-small-base" in plan.reason

    def test_seven_b_base_galore_no_warning(self) -> None:
        from dlm.hardware.plan import resolve

        tc = TrainingConfig(optimizer="galore_adamw")
        plan = resolve(tc, self._make_caps(), seq_len=2048, base_params=7_000_000_000)
        assert "warn=galore-small-base" not in plan.reason
        assert "optim=galore_adamw" in plan.reason

    def test_small_base_plain_adamw_no_warning(self) -> None:
        from dlm.hardware.plan import resolve

        tc = TrainingConfig(optimizer="adamw_torch")
        plan = resolve(tc, self._make_caps(), seq_len=2048, base_params=135_000_000)
        # `adamw_torch` is the default and doesn't appear in reason at all.
        assert "galore" not in plan.reason

    def test_dora_adapter_appears_in_reason(self) -> None:
        from dlm.hardware.plan import resolve

        tc = TrainingConfig(adapter="dora")
        plan = resolve(tc, self._make_caps(), seq_len=2048, base_params=1_500_000_000)
        assert "adapter=dora" in plan.reason
