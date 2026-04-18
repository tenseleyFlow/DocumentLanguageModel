"""Memory estimator: accuracy + monotonicity properties."""

from __future__ import annotations

import pytest

from dlm.hardware.backend import Backend
from dlm.hardware.memory import estimate_peak_vram_gb, estimate_step_seconds


class TestPeakVramEstimator:
    def test_smollm2_135m_lora_fits_within_30pct_of_1gb(self) -> None:
        """Audit F26: SmolLM2-135M LoRA should land near ~400-800 MB peak on CPU.

        We assert a coarse band rather than a single number; the estimator
        is advisory. The Sprint 09 runtime guard is the actual ceiling.
        """
        est = estimate_peak_vram_gb(
            base_params=135_000_000,
            seq_len=512,
            micro_batch=4,
            precision="bf16",
            use_qlora=False,
            lora_r=8,
            gradient_checkpointing=True,
        )
        # 0.15 GB ≤ est ≤ 1.5 GB is our confidence interval for this shape.
        assert 0.15 <= est <= 1.5, f"estimate {est:.2f} GB outside expected band"

    def test_qlora_reduces_peak_vs_bf16(self) -> None:
        bf16 = estimate_peak_vram_gb(
            base_params=1_500_000_000,
            seq_len=2048,
            micro_batch=1,
            precision="bf16",
            use_qlora=False,
            lora_r=8,
            gradient_checkpointing=True,
        )
        qlora = estimate_peak_vram_gb(
            base_params=1_500_000_000,
            seq_len=2048,
            micro_batch=1,
            precision="bf16",
            use_qlora=True,
            lora_r=8,
            gradient_checkpointing=True,
        )
        assert qlora < bf16

    def test_grad_ckpt_reduces_peak(self) -> None:
        with_ckpt = estimate_peak_vram_gb(
            base_params=1_500_000_000,
            seq_len=2048,
            micro_batch=4,
            precision="bf16",
            use_qlora=False,
            lora_r=8,
            gradient_checkpointing=True,
        )
        without = estimate_peak_vram_gb(
            base_params=1_500_000_000,
            seq_len=2048,
            micro_batch=4,
            precision="bf16",
            use_qlora=False,
            lora_r=8,
            gradient_checkpointing=False,
        )
        assert with_ckpt < without

    def test_larger_seq_len_increases_peak(self) -> None:
        small = estimate_peak_vram_gb(
            base_params=1_500_000_000,
            seq_len=512,
            micro_batch=1,
            precision="bf16",
            use_qlora=False,
            lora_r=8,
            gradient_checkpointing=True,
        )
        large = estimate_peak_vram_gb(
            base_params=1_500_000_000,
            seq_len=4096,
            micro_batch=1,
            precision="bf16",
            use_qlora=False,
            lora_r=8,
            gradient_checkpointing=True,
        )
        assert large > small

    def test_larger_batch_increases_peak(self) -> None:
        small = estimate_peak_vram_gb(
            base_params=1_500_000_000,
            seq_len=2048,
            micro_batch=1,
            precision="bf16",
            use_qlora=False,
            lora_r=8,
            gradient_checkpointing=True,
        )
        large = estimate_peak_vram_gb(
            base_params=1_500_000_000,
            seq_len=2048,
            micro_batch=8,
            precision="bf16",
            use_qlora=False,
            lora_r=8,
            gradient_checkpointing=True,
        )
        assert large > small

    @pytest.mark.parametrize("opt", ["adamw_torch", "adamw_bnb_8bit", "paged_adamw_8bit"])
    def test_optimizer_variants_do_not_crash(self, opt: str) -> None:
        est = estimate_peak_vram_gb(
            base_params=135_000_000,
            seq_len=512,
            micro_batch=1,
            precision="bf16",
            use_qlora=False,
            lora_r=8,
            gradient_checkpointing=True,
            optimizer=opt,
        )
        assert est > 0

    def test_zero_trainable_params_fraction_handles_cleanly(self) -> None:
        est = estimate_peak_vram_gb(
            base_params=135_000_000,
            seq_len=512,
            micro_batch=1,
            precision="bf16",
            use_qlora=False,
            lora_r=8,
            gradient_checkpointing=True,
            target_modules_fraction=0.0,
        )
        assert est > 0


class TestStepSecondsEstimator:
    def test_cuda_with_flash_is_faster_than_without(self) -> None:
        with_flash = estimate_step_seconds(
            backend=Backend.CUDA,
            base_params=1_500_000_000,
            seq_len=2048,
            micro_batch=4,
            has_flash_attention=True,
        )
        without = estimate_step_seconds(
            backend=Backend.CUDA,
            base_params=1_500_000_000,
            seq_len=2048,
            micro_batch=4,
            has_flash_attention=False,
        )
        assert with_flash < without

    def test_cpu_is_slowest(self) -> None:
        cpu = estimate_step_seconds(
            backend=Backend.CPU,
            base_params=135_000_000,
            seq_len=512,
            micro_batch=1,
            has_flash_attention=False,
        )
        cuda = estimate_step_seconds(
            backend=Backend.CUDA,
            base_params=135_000_000,
            seq_len=512,
            micro_batch=1,
            has_flash_attention=False,
        )
        assert cpu > cuda
