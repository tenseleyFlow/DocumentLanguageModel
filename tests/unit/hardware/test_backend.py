"""Backend detection across host shapes."""

from __future__ import annotations

from dlm.hardware.backend import Backend, detect
from tests.fixtures.hardware_mocks import force_cpu, force_cuda, force_mps, force_rocm


class TestDetect:
    def test_cpu_when_nothing_available(self) -> None:
        with force_cpu():
            assert detect() == Backend.CPU

    def test_cuda_when_cuda_available_no_hip(self) -> None:
        with force_cuda():
            assert detect() == Backend.CUDA

    def test_rocm_when_hip_set(self) -> None:
        with force_rocm(hip_version="6.0"):
            assert detect() == Backend.ROCM

    def test_mps_when_only_mps_available(self) -> None:
        with force_mps():
            assert detect() == Backend.MPS
