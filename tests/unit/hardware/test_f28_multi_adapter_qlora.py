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
    adapters = {f"a{i}": AdapterConfig(adapter="qlora") for i in range(num)}
    return TrainingConfig.model_validate({"adapters": adapters})


def _qlora_multi_doc_with_rank(num: int, lora_r: int) -> TrainingConfig:
    """Multi-adapter doc with `num` QLoRA adapters at the given lora_r."""
    adapters = {f"a{i}": AdapterConfig(adapter="qlora", lora_r=lora_r) for i in range(num)}
    return TrainingConfig.model_validate({"adapters": adapters})


class TestF28MultiAdapterQLoraRefusal:
    def test_large_base_high_rank_refused(self) -> None:
        # 7B QLoRA, 3 adapters at r=64 on a 12GB device.
        # base: 7*0.5=3.5 GB; per_adapter: 7*64/64=7 GB (×3=21 GB);
        # activations: 7*2*0.25=3.5 GB → 28 GB > 12*0.85=10.2 GB budget.
        with force_cuda(vram_gb=12.0):
            caps = replace(probe(), has_bitsandbytes=True)
        with pytest.raises(ResolutionError, match="Multi-adapter QLoRA"):
            check_refusals(
                _qlora_multi_doc_with_rank(3, 64),
                caps,
                base_params=7_000_000_000,
                num_adapters=3,
            )

    def test_error_message_points_to_adapter_lora_alternative(self) -> None:
        with force_cuda(vram_gb=12.0):
            caps = replace(probe(), has_bitsandbytes=True)
        with pytest.raises(ResolutionError) as exc_info:
            check_refusals(
                _qlora_multi_doc_with_rank(3, 64),
                caps,
                base_params=7_000_000_000,
                num_adapters=3,
            )
        message = str(exc_info.value)
        assert "adapter: lora" in message
        assert "reduce the number of adapters" in message

    def test_error_message_names_offending_adapters(self) -> None:
        """Audit-07 M7/N3: refusal lists which adapters triggered it."""
        with force_cuda(vram_gb=12.0):
            caps = replace(probe(), has_bitsandbytes=True)
        with pytest.raises(ResolutionError) as exc_info:
            check_refusals(
                _qlora_multi_doc_with_rank(3, 64),
                caps,
                base_params=7_000_000_000,
                num_adapters=3,
            )
        message = str(exc_info.value)
        assert "offending adapters" in message
        assert "'a0'" in message

    def test_single_adapter_qlora_not_affected_by_f28(self) -> None:
        # num_adapters=1: F28's `num_adapters > 1` gate skips entirely.
        with force_cuda(vram_gb=4.0):
            caps = replace(probe(), has_bitsandbytes=True)
        flat = TrainingConfig.model_validate({"adapter": "qlora"})
        check_refusals(flat, caps, base_params=1_500_000_000, num_adapters=1)

    def test_multi_adapter_lora_not_refused(self) -> None:
        with force_cuda(vram_gb=4.0):
            caps = replace(probe(), has_bitsandbytes=True)
        adapters = {"a0": AdapterConfig(), "a1": AdapterConfig()}
        lora_multi = TrainingConfig.model_validate({"adapters": adapters})
        # LoRA bypasses QLoRA refusals entirely.
        check_refusals(lora_multi, caps, base_params=1_500_000_000, num_adapters=2)

    def test_small_base_low_rank_multi_qlora_passes(self) -> None:
        """The old formula falsely refused small-base multi-QLoRA.
        The new formula is correctly permissive — 1.5B with r=8 fits in 4GB."""
        with force_cuda(vram_gb=4.0):
            caps = replace(probe(), has_bitsandbytes=True)
        # 1.5B base, r=8, 2 adapters:
        # base 0.75 + per_adapter ~0.19 × 2 + activations 0.75 ≈ 1.9 GB
        # vs 4 × 0.85 = 3.4 GB budget → accepts.
        check_refusals(
            _qlora_multi_doc_with_rank(2, 8),
            caps,
            base_params=1_500_000_000,
            num_adapters=2,
        )

    def test_multi_adapter_qlora_on_large_vram_passes(self) -> None:
        with force_cuda(vram_gb=80.0):  # H100
            caps = replace(probe(), has_bitsandbytes=True)
        # Even 7B + 3 adapters at r=64 (28 GB) fits under 80 × 0.85 = 68.
        check_refusals(
            _qlora_multi_doc_with_rank(3, 64),
            caps,
            base_params=7_000_000_000,
            num_adapters=3,
        )


class TestEffectiveAdapter:
    def test_mixed_multi_adapter_refusal_only_counts_qlora_adapters(self) -> None:
        """Audit-07 M7: mixed doc with one QLoRA + many LoRA doesn't
        get charged the per-adapter VRAM for LoRAs. The formula counts
        only QLoRA-typed adapters in the per-adapter budget line."""
        with force_cuda(vram_gb=12.0):
            caps = replace(probe(), has_bitsandbytes=True)
        # 7B base, 1 QLoRA + 2 LoRA at r=64. Only the 1 QLoRA counts:
        # base 3.5 + per_adapter 7 × 1 + activations 3.5 = 14 GB vs
        # 12 × 0.85 = 10.2 GB budget → refuses (with 1-adapter charge).
        mixed = TrainingConfig.model_validate(
            {
                "adapters": {
                    "qlora_one": {"adapter": "qlora", "lora_r": 64},
                    "lora_a": {"adapter": "lora", "lora_r": 64},
                    "lora_b": {"adapter": "lora", "lora_r": 64},
                },
            }
        )
        with pytest.raises(ResolutionError, match="Multi-adapter QLoRA"):
            check_refusals(mixed, caps, base_params=7_000_000_000, num_adapters=3)

    def test_mixed_adapter_error_names_only_qlora_offenders(self) -> None:
        with force_cuda(vram_gb=12.0):
            caps = replace(probe(), has_bitsandbytes=True)
        mixed = TrainingConfig.model_validate(
            {
                "adapters": {
                    "qlora_one": {"adapter": "qlora", "lora_r": 64},
                    "lora_a": {"adapter": "lora", "lora_r": 64},
                    "lora_b": {"adapter": "lora", "lora_r": 64},
                },
            }
        )
        with pytest.raises(ResolutionError) as exc_info:
            check_refusals(mixed, caps, base_params=7_000_000_000, num_adapters=3)
        message = str(exc_info.value)
        assert "qlora_one" in message
        assert "lora_a" not in message
        assert "lora_b" not in message
