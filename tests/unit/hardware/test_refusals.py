"""Refusal matrix — hard-failure paths for invalid configurations."""

from __future__ import annotations

import pytest

from dlm.doc.schema import TrainingConfig
from dlm.hardware.capabilities import probe
from dlm.hardware.refusals import CPU_PARAM_BUDGET, ResolutionError, check_refusals
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

    def test_qlora_on_rocm_refused_with_sprint_22_pointer(self) -> None:
        with force_rocm():
            caps = probe()
        with pytest.raises(ResolutionError, match="Sprint 22"):
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
