"""Verify hardware_mocks flip torch attributes as advertised."""

from __future__ import annotations

import torch

from tests.fixtures.hardware_mocks import force_cpu, force_cuda, force_mps, force_rocm


class TestForceCuda:
    def test_reports_cuda_available(self) -> None:
        with force_cuda(sm=(8, 6), vram_gb=16.0):
            assert torch.cuda.is_available() is True
            assert torch.cuda.get_device_capability() == (8, 6)
            assert torch.backends.mps.is_available() is False
            assert torch.version.hip is None

    def test_mem_get_info_reports_requested_vram(self) -> None:
        with force_cuda(vram_gb=12.0):
            free, total = torch.cuda.mem_get_info()
            assert free == int(12.0 * (1024**3))
            assert total == free

    def test_restores_state_after_context(self) -> None:
        pre = torch.cuda.is_available()
        with force_cuda():
            assert torch.cuda.is_available() is True
        assert torch.cuda.is_available() is pre


class TestForceMps:
    def test_reports_mps_available(self) -> None:
        with force_mps():
            assert torch.backends.mps.is_available() is True
            assert torch.backends.mps.is_built() is True
            assert torch.cuda.is_available() is False
            assert torch.version.hip is None


class TestForceCpu:
    def test_reports_nothing_available(self) -> None:
        with force_cpu():
            assert torch.cuda.is_available() is False
            assert torch.backends.mps.is_available() is False
            assert torch.version.hip is None


class TestForceRocm:
    def test_reports_hip_version_and_no_mps(self) -> None:
        with force_rocm(hip_version="6.0"):
            # On ROCm, torch.cuda.is_available() is True (hip impersonates cuda)
            assert torch.cuda.is_available() is True
            assert torch.version.hip == "6.0"
            assert torch.backends.mps.is_available() is False


class TestNesting:
    def test_inner_context_overrides_outer(self) -> None:
        with force_cpu():
            assert torch.cuda.is_available() is False
            with force_cuda():
                assert torch.cuda.is_available() is True
            # After inner exits, outer context restored.
            assert torch.cuda.is_available() is False
