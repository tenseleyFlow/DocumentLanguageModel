"""End-to-end doctor() + renderer + CLI integration."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from unittest.mock import patch

from typer.testing import CliRunner

from dlm.cli.app import app
from dlm.hardware import DoctorResult, TrainingPlan, doctor, render_text
from dlm.hardware.backend import Backend
from dlm.hardware.capabilities import Capabilities, probe
from tests.fixtures.hardware_mocks import force_cpu, force_cuda, force_mps


class TestDoctorResult:
    def test_returns_capabilities_on_any_backend(self) -> None:
        with force_mps():
            result = doctor()
        assert result.capabilities.backend.value == "mps"
        assert result.plan is not None  # MPS + default config → plan resolves

    def test_refused_plan_records_error(self) -> None:
        from dlm.doc.schema import TrainingConfig

        with force_mps():
            result = doctor(TrainingConfig(adapter="qlora"))
        assert result.plan is None
        assert result.plan_error is not None
        assert "MPS detected" in result.plan_error

    def test_large_mps_base_records_force_only_error(self) -> None:
        with force_mps():
            caps = probe()
        caps = replace(caps, unified_memory_gb=48.0)
        with patch("dlm.hardware.doctor.probe", return_value=caps):
            result = doctor(base_params=24_000_000_000)
        assert result.plan is None
        assert result.plan_error is not None
        assert "Apple Silicon" in result.plan_error
        assert "--force" in result.plan_error


class TestJsonSerialization:
    def test_to_dict_is_valid_json(self) -> None:
        with force_cuda(sm=(8, 9), vram_gb=24.0):
            result = doctor()
        blob = json.dumps(result.to_dict(), default=str)
        loaded = json.loads(blob)
        assert loaded["capabilities"]["backend"] == "cuda"
        assert loaded["capabilities"]["determinism_class"] == "strong"
        assert loaded["plan"] is not None

    def test_determinism_class_surface(self) -> None:
        with force_cpu():
            cpu_result = doctor()
        with force_mps():
            mps_result = doctor()
        with force_cuda():
            cuda_result = doctor()
        assert cpu_result.capabilities.determinism_class == "advisory"
        assert mps_result.capabilities.determinism_class == "best-effort"
        assert cuda_result.capabilities.determinism_class == "strong"


class TestRender:
    def test_text_output_includes_all_sections(self) -> None:
        with force_mps():
            result = doctor()
        text = render_text(result)
        for marker in (
            "Backend:",
            "Torch:",
            "Determinism:",
            "Telemetry:",
            "Platform:",
            "Suggested plan",
        ):
            assert marker in text, f"missing marker {marker!r} in render"

    def test_text_output_on_refused_plan(self) -> None:
        from dlm.doc.schema import TrainingConfig

        with force_mps():
            result = doctor(TrainingConfig(adapter="qlora"))
        text = render_text(result)
        assert "Plan refused:" in text
        assert "MPS" in text

    def test_cuda_render_surfaces_sm_vram_and_cuda_suffix(self) -> None:
        caps = Capabilities(
            backend=Backend.CUDA,
            device_name="RTX 4090",
            sm=(8, 9),
            rocm_arch=None,
            vram_gb=23.9,
            unified_memory_gb=None,
            cpu_cores=16,
            ram_gb=64.0,
            supports_bf16=True,
            supports_fp16=True,
            has_flash_attention=True,
            has_xformers=False,
            has_bitsandbytes=True,
            has_triton=True,
            has_mlx=False,
            torch_version="2.11.0",
            accelerate_version="1.2.0",
            cuda_version="12.4",
            rocm_version=None,
            platform="Linux 6.8",
            determinism_class="strong",
            telemetry_posture={},
        )
        plan = TrainingPlan(
            precision="bf16",
            attn_implementation="flash_attention_2",
            use_qlora=False,
            quant_compute_dtype=None,
            micro_batch_size=2,
            grad_accum=4,
            effective_batch_size=8,
            gradient_checkpointing=False,
            est_peak_vram_gb=7.5,
            est_step_seconds=0.8,
            reason="test",
            world_size=1,
        )
        text = render_text(DoctorResult(capabilities=caps, plan=plan, plan_error=None))
        assert "SM 8.9" in text
        assert "23.9 GB VRAM free" in text
        assert "Torch:          2.11.0 (CUDA 12.4)" in text

    def test_rocm_render_surfaces_arch_suffix_and_qlora_summary(self) -> None:
        caps = Capabilities(
            backend=Backend.ROCM,
            device_name="Radeon 7900 XTX",
            sm=(11, 0),
            rocm_arch="gfx1100",
            vram_gb=15.5,
            unified_memory_gb=None,
            cpu_cores=16,
            ram_gb=64.0,
            supports_bf16=True,
            supports_fp16=True,
            has_flash_attention=False,
            has_xformers=False,
            has_bitsandbytes=False,
            has_triton=True,
            has_mlx=False,
            torch_version="2.11.0",
            accelerate_version=None,
            cuda_version=None,
            rocm_version="6.0",
            platform="Linux 6.8",
            determinism_class="best-effort",
            telemetry_posture={},
        )
        plan = TrainingPlan(
            precision="bf16",
            attn_implementation="sdpa",
            use_qlora=True,
            quant_compute_dtype="bf16",
            micro_batch_size=1,
            grad_accum=8,
            effective_batch_size=8,
            gradient_checkpointing=True,
            est_peak_vram_gb=8.2,
            est_step_seconds=1.2,
            reason="test",
            world_size=1,
        )
        text = render_text(DoctorResult(capabilities=caps, plan=plan, plan_error=None))
        assert "arch gfx1100" in text
        assert "Torch:          2.11.0 (ROCm 6.0)" in text
        assert "adapter:         qlora (4-bit NF4, compute bf16)" in text


class TestCliDoctor:
    def test_cli_human_output_works(self) -> None:
        runner = CliRunner()
        # Exit code + rough output shape; underlying backend is whatever
        # the test host has.
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0, result.output
        assert "Backend:" in result.output
        assert "Suggested plan" in result.output or "Plan refused" in result.output

    def test_cli_json_output_is_valid_json(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["doctor", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "capabilities" in payload
        assert "plan" in payload
        assert payload["capabilities"]["determinism_class"] in {
            "strong",
            "best-effort",
            "advisory",
        }

    def test_dlm_doctor_subprocess_exits_zero(self) -> None:
        """Belt-and-braces: `python -m dlm doctor` on the real process."""
        result = subprocess.run(
            [sys.executable, "-m", "dlm", "doctor"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "Determinism:" in result.stdout
