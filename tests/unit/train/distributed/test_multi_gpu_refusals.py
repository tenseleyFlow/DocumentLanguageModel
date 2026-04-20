"""Multi-GPU refusal matrix — MPS/CPU/ROCm + heterogeneous CUDA."""

from __future__ import annotations

import pytest

from dlm.doc.schema import TrainingConfig
from dlm.hardware.capabilities import probe
from dlm.hardware.refusals import (
    ResolutionError,
    assert_homogeneous_cuda,
    check_multi_gpu_refusals,
)
from tests.fixtures.hardware_mocks import force_cpu, force_cuda, force_mps, force_rocm


def _cfg(**overrides: object) -> TrainingConfig:
    base = {"adapter": "lora", "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05}
    base.update(overrides)
    return TrainingConfig(**base)  # type: ignore[arg-type]


class TestCheckMultiGpuRefusals:
    def test_world_size_1_is_noop(self) -> None:
        with force_mps():
            caps = probe()
        check_multi_gpu_refusals(caps, world_size=1)

    def test_refuses_mps(self) -> None:
        with force_mps():
            caps = probe()
        with pytest.raises(ResolutionError, match="Apple Silicon"):
            check_multi_gpu_refusals(caps, world_size=2)

    def test_refuses_cpu(self) -> None:
        with force_cpu():
            caps = probe()
        with pytest.raises(ResolutionError, match="CPU"):
            check_multi_gpu_refusals(caps, world_size=4)

    def test_refuses_rocm(self) -> None:
        with force_rocm():
            caps = probe()
        with pytest.raises(ResolutionError, match="ROCm"):
            check_multi_gpu_refusals(caps, world_size=2)

    def test_cuda_passes_through(self) -> None:
        with force_cuda():
            caps = probe()
        # No refusal on CUDA multi-GPU; heterogeneity check is a
        # separate surface the caller drives with device properties.
        check_multi_gpu_refusals(caps, world_size=2)


class TestMultiGpuQLoRACudaAllowed:
    """Audit-08 M4: the one path Phase 5 promises works has to be tested."""

    def test_multi_gpu_qlora_cuda_passes_multi_gpu_refusals(self) -> None:
        """`check_multi_gpu_refusals` doesn't block CUDA multi-GPU."""
        with force_cuda():
            caps = probe()
        check_multi_gpu_refusals(caps, world_size=2)
        check_multi_gpu_refusals(caps, world_size=8)

    def test_qlora_refusal_matrix_under_cuda_does_not_refuse(self) -> None:
        """CUDA+bnb+QLoRA+multi-GPU passes both refusal gates.

        Covers the audit-08 M4 concern: without this test, "QLoRA +
        multi-GPU CUDA permitted" could regress silently if someone
        tightened the refusal matrix. Pairs `check_multi_gpu_refusals`
        (rejects MPS/CPU/ROCm) with `check_refusals` (rejects QLoRA
        without bnb) and asserts neither fires on the allowed combo.
        """
        from dataclasses import replace

        from dlm.hardware.refusals import check_refusals

        with force_cuda():
            caps = probe()
        # Pretend bnb is installed so the QLoRA gate doesn't trip on
        # test-env missing-bnb (dev laptops don't have it).
        caps_with_bnb = replace(caps, has_bitsandbytes=True)

        # Neither refusal fires on the CUDA+bnb+QLoRA+multi-GPU combo.
        check_multi_gpu_refusals(caps_with_bnb, world_size=2)
        check_refusals(
            _cfg(adapter="qlora"),
            caps_with_bnb,
            base_params=135_000_000,
            num_adapters=1,
        )


class TestAssertHomogeneousCuda:
    def test_homogeneous_passes(self) -> None:
        assert_homogeneous_cuda([(8, 0), (8, 0)])

    def test_single_device_passes(self) -> None:
        assert_homogeneous_cuda([(8, 9)])

    def test_mixed_sm_refused(self) -> None:
        with pytest.raises(ResolutionError, match="Heterogeneous"):
            assert_homogeneous_cuda([(8, 0), (7, 5)])

    def test_none_entries_ignored(self) -> None:
        # An unknown SM doesn't participate in the homogeneity check.
        assert_homogeneous_cuda([(8, 0), None, (8, 0)])

    def test_all_none_passes(self) -> None:
        # Edge case: no arch info for any device → can't refuse.
        assert_homogeneous_cuda([None, None])
