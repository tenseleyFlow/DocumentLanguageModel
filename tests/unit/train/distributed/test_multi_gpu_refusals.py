"""Multi-GPU refusal matrix — MPS/CPU/ROCm + heterogeneous CUDA."""

from __future__ import annotations

import pytest

from dlm.hardware.capabilities import probe
from dlm.hardware.refusals import (
    ResolutionError,
    assert_homogeneous_cuda,
    check_multi_gpu_refusals,
)
from tests.fixtures.hardware_mocks import force_cpu, force_cuda, force_mps, force_rocm


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
