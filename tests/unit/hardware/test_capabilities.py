"""Capabilities probe — per-backend fields, determinism class, telemetry posture."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dlm.hardware.backend import Backend
from dlm.hardware.capabilities import _accelerate_version, _rocm_arch_supports_bf16, probe
from tests.fixtures.hardware_mocks import force_cpu, force_cuda, force_mps, force_rocm


class TestProbeCuda:
    def test_cuda_caps_populate_device_fields(self) -> None:
        with force_cuda(sm=(8, 9), vram_gb=24.0, device_name="NVIDIA RTX 4090"):
            caps = probe()
        assert caps.backend == Backend.CUDA
        assert caps.device_name == "NVIDIA RTX 4090"
        assert caps.sm == (8, 9)
        assert caps.vram_gb is not None
        assert 23.5 <= caps.vram_gb <= 24.5
        assert caps.unified_memory_gb is None
        assert caps.supports_bf16 is True
        assert caps.supports_fp16 is True
        assert caps.determinism_class == "strong"

    def test_cuda_pre_ampere_no_bf16(self) -> None:
        with force_cuda(sm=(7, 5)):
            caps = probe()
        assert caps.supports_bf16 is False
        # flash_attn gated on SM>=8.0 regardless of package availability
        assert caps.has_flash_attention is False

    def test_cuda_sm_probe_failure_yields_unknown_sm(self) -> None:
        with force_cuda():
            with patch("torch.cuda.get_device_capability", side_effect=RuntimeError("boom")):
                caps = probe()
        assert caps.sm is None

    def test_cuda_vram_probe_failure_yields_unknown_vram(self) -> None:
        with force_cuda():
            with patch("torch.cuda.mem_get_info", side_effect=RuntimeError("boom")):
                caps = probe()
        assert caps.vram_gb is None

    def test_cuda_flash_attention_true_when_package_and_sm_supported(self) -> None:
        with (
            patch("dlm.hardware.capabilities._module_available", lambda name: name == "flash_attn"),
            force_cuda(sm=(8, 0)),
        ):
            caps = probe()
        assert caps.has_flash_attention is True


class TestProbeRocm:
    def test_rocm_reports_hip_version(self) -> None:
        with force_rocm(hip_version="6.0"):
            caps = probe()
        assert caps.backend == Backend.ROCM
        assert caps.rocm_version == "6.0"
        assert caps.cuda_version is None
        assert caps.determinism_class == "best-effort"
        assert caps.has_flash_attention is False

    def test_rocm_arch_probe_failure_yields_unknown_arch(self) -> None:
        with force_rocm():
            with patch("torch.cuda.get_device_properties", side_effect=RuntimeError("boom")):
                caps = probe()
        assert caps.rocm_arch is None

    def test_rocm_arch_probe_missing_name_yields_unknown_arch(self) -> None:
        with force_rocm():
            with patch(
                "torch.cuda.get_device_properties", return_value=SimpleNamespace(name="AMD")
            ):
                caps = probe()
        assert caps.rocm_arch is None


class TestProbeMps:
    def test_mps_caps(self) -> None:
        with force_mps():
            caps = probe()
        assert caps.backend == Backend.MPS
        assert caps.supports_bf16 is False  # conservative default
        assert caps.supports_fp16 is True
        assert caps.unified_memory_gb is not None
        assert caps.vram_gb is None
        assert caps.determinism_class == "best-effort"
        assert caps.has_flash_attention is False

    def test_mps_never_reports_flash_attention(self) -> None:
        with (
            patch("dlm.hardware.capabilities._module_available", lambda name: name == "flash_attn"),
            force_mps(),
        ):
            caps = probe()
        assert caps.has_flash_attention is False


class TestMlxAvailability:
    def test_non_mps_never_reports_mlx(self) -> None:
        # Off-Apple hosts: has_mlx is False regardless of dist metadata
        # (Sprint 21). The probe won't consult importlib on CUDA/CPU.
        with force_cuda(sm=(8, 0)):
            caps = probe()
        assert caps.has_mlx is False
        with force_cpu():
            caps = probe()
        assert caps.has_mlx is False

    def test_mps_reports_mlx_when_both_modules_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate both mlx + mlx_lm available.
        from dlm.hardware import capabilities as caps_mod

        real_avail = caps_mod._module_available

        def fake_available(name: str) -> bool:
            if name in ("mlx", "mlx_lm"):
                return True
            return real_avail(name)

        monkeypatch.setattr(caps_mod, "_module_available", fake_available)
        with force_mps():
            caps = probe()
        assert caps.has_mlx is True

    def test_mps_reports_no_mlx_when_mlx_lm_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from dlm.hardware import capabilities as caps_mod

        real_avail = caps_mod._module_available

        def fake_available(name: str) -> bool:
            if name == "mlx":
                return True
            if name == "mlx_lm":
                return False
            return real_avail(name)

        monkeypatch.setattr(caps_mod, "_module_available", fake_available)
        with force_mps():
            caps = probe()
        assert caps.has_mlx is False


class TestProbeCpu:
    def test_cpu_advisory_determinism(self) -> None:
        with force_cpu():
            caps = probe()
        assert caps.backend == Backend.CPU
        assert caps.determinism_class == "advisory"
        assert caps.vram_gb is None
        assert caps.unified_memory_gb is None
        assert caps.supports_bf16 is False
        assert caps.has_flash_attention is False


class TestTelemetryPosture:
    def test_reports_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        with force_cpu():
            caps = probe()
        assert caps.telemetry_posture["HF_HUB_DISABLE_TELEMETRY"] == "1"
        assert caps.telemetry_posture["DO_NOT_TRACK"] == "1"
        assert "wandb_installed" in caps.telemetry_posture
        assert "python" in caps.telemetry_posture

    def test_unset_vars_show_placeholder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
        monkeypatch.delenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", raising=False)
        with force_cpu():
            caps = probe()
        assert caps.telemetry_posture["HF_HUB_DISABLE_TELEMETRY"] == "<unset>"
        assert caps.telemetry_posture["DO_NOT_TRACK"] == "<unset>"


class TestCoverageEdges:
    def test_rocm_arch_none_is_not_bf16_capable(self) -> None:
        assert _rocm_arch_supports_bf16(None) is False

    def test_accelerate_version_missing_returns_none(self) -> None:
        from importlib.metadata import PackageNotFoundError

        with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            assert _accelerate_version() is None
