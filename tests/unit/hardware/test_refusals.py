"""Refusal matrix — hard-failure paths for invalid configurations."""

from __future__ import annotations

import pytest

from dlm.doc.schema import AdapterConfig, TrainingConfig
from dlm.hardware.capabilities import probe
from dlm.hardware.refusals import (
    CPU_PARAM_BUDGET,
    ResolutionError,
    _avg_lora_r,
    _qlora_adapter_count,
    _qlora_adapter_names,
    assert_homogeneous_cuda,
    check_multi_gpu_refusals,
    check_refusals,
)
from tests.fixtures.hardware_mocks import force_cpu, force_cuda, force_mps, force_rocm


def _cfg(**overrides: object) -> TrainingConfig:
    data: dict[str, object] = {"adapter": "lora"}
    data.update(overrides)
    return TrainingConfig.model_validate(data)


class TestQloraRefusals:
    def test_qlora_on_mps_refused(self) -> None:
        with force_mps():
            caps = probe()
        with pytest.raises(ResolutionError, match="MPS detected"):
            check_refusals(_cfg(adapter="qlora"), caps, base_params=1_500_000_000)

    def test_qlora_on_cpu_refused(self) -> None:
        with force_cpu():
            caps = probe()
        with pytest.raises(ResolutionError, match="CPU detected"):
            check_refusals(_cfg(adapter="qlora"), caps, base_params=135_000_000)

    def test_qlora_on_rocm_refused_permanently(self) -> None:
        with force_rocm():
            caps = probe()
        with pytest.raises(ResolutionError, match="bitsandbytes.*ROCm"):
            check_refusals(_cfg(adapter="qlora"), caps, base_params=1_500_000_000)

    def test_qlora_on_cuda_without_bnb_refused(self) -> None:
        # bitsandbytes isn't installed locally; probe reports has_bitsandbytes=False.
        with force_cuda():
            caps = probe()
        if caps.has_bitsandbytes:
            pytest.skip("bitsandbytes actually installed; can't assert refusal")
        with pytest.raises(ResolutionError, match="bitsandbytes"):
            check_refusals(_cfg(adapter="qlora"), caps, base_params=1_500_000_000)


class TestCpuParameterBudget:
    def test_cpu_threshold_is_200m(self) -> None:
        """Audit F24: the budget is 200M, not 3B."""
        assert CPU_PARAM_BUDGET == 200_000_000

    def test_cpu_with_over_budget_refused(self) -> None:
        with force_cpu():
            caps = probe()
        with pytest.raises(ResolutionError, match="200M"):
            check_refusals(_cfg(), caps, base_params=CPU_PARAM_BUDGET + 1)

    def test_cpu_at_budget_passes(self) -> None:
        with force_cpu():
            caps = probe()
        # Exactly at the budget is allowed (strict > in the check).
        check_refusals(_cfg(), caps, base_params=CPU_PARAM_BUDGET)

    def test_cpu_below_budget_passes(self) -> None:
        with force_cpu():
            caps = probe()
        check_refusals(_cfg(), caps, base_params=135_000_000)

    def test_force_bypasses_cpu_budget(self) -> None:
        with force_cpu():
            caps = probe()
        check_refusals(_cfg(), caps, base_params=1_500_000_000, force=True)

    def test_force_does_not_bypass_qlora_refusal(self) -> None:
        """Even with --force, QLoRA on MPS can't work (no bitsandbytes)."""
        with force_mps():
            caps = probe()
        with pytest.raises(ResolutionError):
            check_refusals(_cfg(adapter="qlora"), caps, base_params=1_000_000, force=True)


class TestLoraOnAllBackends:
    """LoRA should be accepted on every backend (subject to CPU budget)."""

    def test_lora_on_cuda(self) -> None:
        with force_cuda():
            caps = probe()
        check_refusals(_cfg(adapter="lora"), caps, base_params=1_500_000_000)

    def test_lora_on_mps(self) -> None:
        with force_mps():
            caps = probe()
        check_refusals(_cfg(adapter="lora"), caps, base_params=1_500_000_000)

    def test_lora_on_cpu_under_budget(self) -> None:
        with force_cpu():
            caps = probe()
        check_refusals(_cfg(adapter="lora"), caps, base_params=100_000_000)


class TestMultiGpuRefusals:
    def test_world_size_one_is_allowed_on_any_backend(self) -> None:
        with force_cpu():
            caps = probe()
        check_multi_gpu_refusals(caps, world_size=1)

    @pytest.mark.parametrize(
        ("ctx", "expected"),
        [
            (force_mps, "Apple Silicon"),
            (force_cpu, "CPU"),
            (force_rocm, "ROCm"),
        ],
    )
    def test_unsupported_backends_raise(
        self,
        ctx: object,
        expected: str,
    ) -> None:
        with ctx():
            caps = probe()
        with pytest.raises(ResolutionError, match=expected):
            check_multi_gpu_refusals(caps, world_size=2)

    def test_homogeneous_cuda_accepts_unknown_or_matching_sms(self) -> None:
        assert_homogeneous_cuda([(8, 0), (8, 0), None])

    def test_homogeneous_cuda_refuses_mixed_sms(self) -> None:
        with pytest.raises(ResolutionError, match="Heterogeneous CUDA GPUs"):
            assert_homogeneous_cuda([(8, 0), (7, 5)])


class TestPrivateHelpers:
    def test_flat_training_helpers_use_fallbacks(self) -> None:
        training = _cfg(adapter="qlora", lora_r=32)
        assert _avg_lora_r(training) == 32.0
        assert _qlora_adapter_count(training, fallback=3) == 3
        assert _qlora_adapter_names(training) == []

    def test_multi_adapter_helpers_read_declared_adapters(self) -> None:
        training = TrainingConfig.model_validate(
            {
                "adapters": {
                    "knowledge": AdapterConfig(adapter="qlora", lora_r=8),
                    "style": AdapterConfig(adapter="lora", lora_r=24),
                    "tools": AdapterConfig(adapter="qlora", lora_r=16),
                }
            }
        )
        assert _avg_lora_r(training) == pytest.approx((8 + 24 + 16) / 3)
        assert _qlora_adapter_count(training, fallback=1) == 2
        assert _qlora_adapter_names(training) == ["knowledge", "tools"]

    def test_qlora_cuda_without_bnb_refusal_can_be_forced_in_test(self) -> None:
        with force_cuda():
            caps = probe()
        caps = type(caps)(**{**caps.__dict__, "has_bitsandbytes": False})
        with pytest.raises(ResolutionError, match="bitsandbytes"):
            check_refusals(_cfg(adapter="qlora"), caps, base_params=1_500_000_000)
