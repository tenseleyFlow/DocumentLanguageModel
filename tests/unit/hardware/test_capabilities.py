"""Capabilities probe — per-backend fields, determinism class, telemetry posture."""

from __future__ import annotations

import pytest

from dlm.hardware.backend import Backend
from dlm.hardware.capabilities import probe
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


class TestProbeRocm:
    def test_rocm_reports_hip_version(self) -> None:
        with force_rocm(hip_version="6.0"):
            caps = probe()
        assert caps.backend == Backend.ROCM
        assert caps.rocm_version == "6.0"
        assert caps.cuda_version is None
        assert caps.determinism_class == "best-effort"
        assert caps.has_flash_attention is False


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
