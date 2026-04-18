"""End-to-end doctor() + renderer + CLI integration."""

from __future__ import annotations

import json
import subprocess
import sys

from typer.testing import CliRunner

from dlm.cli.app import app
from dlm.hardware import doctor, render_text
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
