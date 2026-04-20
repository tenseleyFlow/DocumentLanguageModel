"""F28: multi-adapter + QLoRA VRAM refusal (Sprint 20b)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dlm.doc.schema import AdapterConfig, TrainingConfig
from dlm.hardware.capabilities import probe
from dlm.hardware.refusals import ResolutionError, check_refusals
from tests.fixtures.hardware_mocks import force_cuda


def _qlora_multi_doc(num: int) -> TrainingConfig:
    """TrainingConfig with `num` QLoRA adapters declared."""
    adapters = {
        f"a{i}": AdapterConfig(adapter="qlora") for i in range(num)
    }
    return TrainingConfig.model_validate({"adapters": adapters})


class TestF28MultiAdapterQLoraRefusal:
    def test_two_adapters_on_small_vram_refused(self) -> None:
        with force_cuda(vram_gb=4.0):
            caps = replace(probe(), has_bitsandbytes=True)
        # 1.5B-param base at 4-bit ≈ 0.75 GB + 2 * 1 GB + ~0.75 GB activ
        # ≈ 3.5 GB > 4 * 0.85 = 3.4 GB budget.
        with pytest.raises(ResolutionError, match="Multi-adapter QLoRA"):
            check_refusals(
                _qlora_multi_doc(2),
                caps,
                base_params=1_500_000_000,
                num_adapters=2,
            )

    def test_error_message_points_to_adapter_lora_alternative(self) -> None:
        with force_cuda(vram_gb=4.0):
            caps = replace(probe(), has_bitsandbytes=True)
        with pytest.raises(ResolutionError) as exc_info:
            check_refusals(
                _qlora_multi_doc(3),
                caps,
                base_params=1_500_000_000,
                num_adapters=3,
            )
        message = str(exc_info.value)
        assert "adapter: lora" in message
        assert "reduce the number of adapters" in message

    def test_single_adapter_qlora_not_affected_by_f28(self) -> None:
        # num_adapters=1 on a small VRAM box: the multi-adapter F28 gate
        # doesn't fire (single-adapter QLoRA is the normal path); other
        # refusals still apply but F28 specifically does not.
        with force_cuda(vram_gb=4.0):
            caps = replace(probe(), has_bitsandbytes=True)
        flat = TrainingConfig.model_validate({"adapter": "qlora"})
        # No raise: the QLoRA checks pass (bnb present) and num_adapters
        # defaults to 1, so F28's `num_adapters > 1` gate skips.
        check_refusals(flat, caps, base_params=1_500_000_000, num_adapters=1)

    def test_multi_adapter_lora_not_refused(self) -> None:
        with force_cuda(vram_gb=4.0):
            caps = replace(probe(), has_bitsandbytes=True)
        adapters = {"a0": AdapterConfig(), "a1": AdapterConfig()}
        lora_multi = TrainingConfig.model_validate({"adapters": adapters})
        # LoRA bypasses QLoRA refusals entirely.
        check_refusals(
            lora_multi, caps, base_params=1_500_000_000, num_adapters=2
        )

    def test_multi_adapter_qlora_on_large_vram_passes(self) -> None:
        with force_cuda(vram_gb=80.0):  # H100
            caps = replace(probe(), has_bitsandbytes=True)
        # 1.5B base → 0.75 + 3*1 + 0.75 ≈ 4.5 GB, well under 80 * 0.85 = 68.
        check_refusals(
            _qlora_multi_doc(3),
            caps,
            base_params=1_500_000_000,
            num_adapters=3,
        )


class TestEffectiveAdapter:
    def test_mixed_multi_adapter_treated_as_qlora_for_refusals(self) -> None:
        """If any declared adapter is QLoRA, F28 applies."""
        with force_cuda(vram_gb=4.0):
            caps = replace(probe(), has_bitsandbytes=True)
        # One LoRA + one QLoRA → the "any qlora" rule still triggers
        # F28's per-adapter math against num_adapters=2.
        mixed = TrainingConfig.model_validate(
            {
                "adapters": {
                    "lora_one": {"adapter": "lora"},
                    "qlora_two": {"adapter": "qlora"},
                },
            }
        )
        with pytest.raises(ResolutionError, match="Multi-adapter QLoRA"):
            check_refusals(
                mixed, caps, base_params=1_500_000_000, num_adapters=2
            )
