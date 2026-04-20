"""ROCm training-plan resolution — Sprint 22.

Covers:

- Arch-aware bf16 probe (`gfx90a` / `gfx942` / `gfx1100` → bf16,
  `gfx1030` RDNA2 / `gfx906` Vega20 → fp16).
- FlashAttention gating: bf16-capable arch + `flash_attn` importable
  → FA2 enabled; bf16-incapable arch always SDPA regardless of
  package presence.
- QLoRA refusal: permanent, no "Sprint 22" pointer.
- `TrainingPlan` picks correct precision + attention on each arch.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dlm.doc.schema import TrainingConfig
from dlm.hardware.backend import Backend
from dlm.hardware.capabilities import probe
from dlm.hardware.plan import resolve
from dlm.hardware.refusals import ResolutionError, check_refusals
from tests.fixtures.hardware_mocks import force_rocm


def _cfg(**overrides: object) -> TrainingConfig:
    base = {"adapter": "lora", "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05}
    base.update(overrides)
    return TrainingConfig(**base)  # type: ignore[arg-type]


class TestRocmBf16Matrix:
    @pytest.mark.parametrize(
        ("arch", "expected"),
        [
            ("gfx90a", True),  # MI200
            ("gfx942", True),  # MI300
            ("gfx1100", True),  # RDNA3 7900 XTX
            ("gfx1101", True),  # RDNA3 7800 XT
            ("gfx1102", True),  # RDNA3 7700 XT
            ("gfx1030", False),  # RDNA2 6900 XT — no bf16
            ("gfx906", False),  # Vega20 — no bf16
            ("gfx908", False),  # CDNA1 MI100 — no bf16
        ],
    )
    def test_bf16_by_arch(self, arch: str, expected: bool) -> None:
        with force_rocm(gcn_arch_name=arch):
            caps = probe()
        assert caps.backend == Backend.ROCM
        assert caps.rocm_arch == arch
        assert caps.supports_bf16 is expected

    def test_bf16_plan_picks_bf16_on_rdna3(self) -> None:
        with force_rocm(gcn_arch_name="gfx1100"):
            caps = probe()
        plan = resolve(_cfg(), caps, base_params=135_000_000, seq_len=512)
        assert plan.precision == "bf16"

    def test_bf16_plan_falls_back_to_fp16_on_rdna2(self) -> None:
        with force_rocm(gcn_arch_name="gfx1030"):
            caps = probe()
        plan = resolve(_cfg(), caps, base_params=135_000_000, seq_len=512)
        assert plan.precision == "fp16"

    def test_gcn_arch_strips_xnack_suffix(self) -> None:
        """ROCm sometimes appends `:sramecc+:xnack-` — we match the bare arch."""
        with force_rocm(gcn_arch_name="gfx90a:sramecc+:xnack-"):
            caps = probe()
        assert caps.rocm_arch == "gfx90a"
        assert caps.supports_bf16 is True


class TestRocmFlashAttention:
    def test_fa2_enabled_when_package_present_and_arch_ok(self) -> None:
        with (
            patch(
                "dlm.hardware.capabilities._module_available",
                lambda name: name == "flash_attn",
            ),
            force_rocm(gcn_arch_name="gfx90a"),
        ):
            caps = probe()
        assert caps.has_flash_attention is True

    def test_fa2_disabled_when_package_absent(self) -> None:
        with (
            patch("dlm.hardware.capabilities._module_available", lambda name: False),
            force_rocm(gcn_arch_name="gfx1100"),
        ):
            caps = probe()
        assert caps.has_flash_attention is False

    def test_fa2_disabled_on_rdna2_even_with_package(self) -> None:
        """bf16-incapable arch never gets FA2, regardless of `flash_attn` install."""
        with (
            patch(
                "dlm.hardware.capabilities._module_available",
                lambda name: name == "flash_attn",
            ),
            force_rocm(gcn_arch_name="gfx1030"),
        ):
            caps = probe()
        assert caps.has_flash_attention is False


class TestRocmQLoRARefusal:
    def test_qlora_on_rocm_refuses_with_bitsandbytes_explanation(self) -> None:
        with force_rocm():
            caps = probe()
        with pytest.raises(ResolutionError) as exc_info:
            check_refusals(_cfg(adapter="qlora"), caps, base_params=1_500_000_000)
        msg = str(exc_info.value)
        assert "bitsandbytes" in msg
        assert "ROCm" in msg
        # Sprint 22 landed; the message must not promise future work.
        assert "Sprint 22" not in msg
        assert "wait for" not in msg.lower()

    def test_lora_on_rocm_passes(self) -> None:
        with force_rocm():
            caps = probe()
        # Should not raise.
        check_refusals(_cfg(adapter="lora"), caps, base_params=1_500_000_000)


class TestRocmPlanIntegration:
    def test_tier2_happy_path_rdna3(self) -> None:
        with force_rocm(gcn_arch_name="gfx1100", vram_gb=24.0):
            caps = probe()
        plan = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048)
        assert plan.precision == "bf16"
        # Without flash_attn installed locally, SDPA is the expected fallback.
        assert plan.attn_implementation in ("sdpa", "flash_attention_2")
        assert plan.use_qlora is False
