"""Disk preflight — estimate math + free-space gating."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dlm.base_models import BASE_MODELS
from dlm.train.disk_preflight import estimate_checkpoint_bytes, preflight_disk
from dlm.train.errors import DiskSpaceError


def _plan() -> SimpleNamespace:
    """Minimal TrainingPlan stand-in; only attrs disk_preflight reads."""
    return SimpleNamespace(
        precision="bf16",
        attn_implementation="sdpa",
        use_qlora=False,
        quant_compute_dtype=None,
        micro_batch_size=1,
        grad_accum=2,
        effective_batch_size=2,
        gradient_checkpointing=False,
        est_peak_vram_gb=4.0,
        est_step_seconds=1.0,
        reason="test",
    )


class TestEstimate:
    def test_tiny_model_hits_floor(self) -> None:
        """SmolLM2-135M: adapter floor (50 MB) + optimizer (~125 MB)."""
        est = estimate_checkpoint_bytes(BASE_MODELS["smollm2-135m"], _plan())
        # Floor: 50MB adapter × 2.5 (optimizer) + 10MB log reserve = ~185MB.
        assert 100_000_000 <= est <= 300_000_000

    def test_larger_model_larger_estimate(self) -> None:
        small = estimate_checkpoint_bytes(BASE_MODELS["qwen2.5-0.5b"], _plan())
        big = estimate_checkpoint_bytes(BASE_MODELS["qwen2.5-3b"], _plan())
        assert big >= small


class TestPreflight:
    def test_ample_disk_passes(self, tmp_path: Path) -> None:
        """Default temp path has GB of headroom on CI runners."""
        preflight_disk(tmp_path, BASE_MODELS["smollm2-135m"], _plan())

    def test_insufficient_disk_raises(self, tmp_path: Path) -> None:
        """Mock `disk_usage` to return just 1 MB free."""
        fake_usage = SimpleNamespace(total=1_000_000, used=0, free=1_000_000)
        with (
            patch("dlm.train.disk_preflight.shutil.disk_usage", return_value=fake_usage),
            pytest.raises(DiskSpaceError, match="insufficient"),
        ):
            preflight_disk(tmp_path, BASE_MODELS["smollm2-135m"], _plan())

    def test_negative_safety_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="safety"):
            preflight_disk(tmp_path, BASE_MODELS["smollm2-135m"], _plan(), safety=0)

    def test_error_carries_numeric_fields(self, tmp_path: Path) -> None:
        fake_usage = SimpleNamespace(total=1_000_000, used=0, free=1_000_000)
        with (
            patch("dlm.train.disk_preflight.shutil.disk_usage", return_value=fake_usage),
            pytest.raises(DiskSpaceError) as excinfo,
        ):
            preflight_disk(tmp_path, BASE_MODELS["smollm2-135m"], _plan())
        assert excinfo.value.free_bytes == 1_000_000
        assert excinfo.value.required_bytes > excinfo.value.free_bytes
