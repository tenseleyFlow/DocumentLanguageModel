"""TrainingPlan resolver behavior across backends."""

from __future__ import annotations

from dlm.doc.schema import TrainingConfig
from dlm.hardware.capabilities import probe
from dlm.hardware.plan import resolve
from tests.fixtures.hardware_mocks import force_cpu, force_cuda, force_mps


def _cfg(**overrides: object) -> TrainingConfig:
    data: dict[str, object] = {}
    data.update(overrides)
    return TrainingConfig.model_validate(data)


class TestPrecisionPicker:
    def test_ampere_gets_bf16(self) -> None:
        with force_cuda(sm=(8, 0)):
            caps = probe()
        plan = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048)
        assert plan.precision == "bf16"

    def test_pre_ampere_gets_fp16(self) -> None:
        with force_cuda(sm=(7, 5)):
            caps = probe()
        plan = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048)
        assert plan.precision == "fp16"

    def test_mps_gets_fp32(self) -> None:
        # MPS fp16 attention kernels produce NaN LoRA weights on tiny-data
        # runs (see .docs/bugs/01-nan-adapter-on-mps.md). Pinned to fp32
        # until PyTorch's MPS kernels land a fix.
        with force_mps():
            caps = probe()
        plan = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048)
        assert plan.precision == "fp32"

    def test_mps_fp16_override_honored(self, caplog: object) -> None:
        # Frontmatter `training.precision: fp16` on MPS wins over the
        # default fp32 pin (see .docs/bugs/01-nan-adapter-on-mps.md —
        # user is opting in to the known NaN risk for memory headroom).
        import logging

        with force_mps():
            caps = probe()
        with caplog.at_level(logging.WARNING, logger="dlm.hardware.plan"):  # type: ignore[attr-defined]
            plan = resolve(
                _cfg(precision="fp16"),
                caps,
                base_params=8_000_000_000,
                seq_len=2048,
                force=True,
            )
        assert plan.precision == "fp16"
        # The caller must see the risk explicitly — silent fp16 on MPS
        # is what caused the original bug.
        messages = " ".join(r.message for r in caplog.records)  # type: ignore[attr-defined]
        assert "fp16 on MPS" in messages
        assert "NaN" in messages

    def test_mps_bf16_override_silent(self, caplog: object) -> None:
        # bf16 on MPS is user-asserted; no warning, since bf16 doesn't
        # have the fp16 attention-kernel issue.
        import logging

        with force_mps():
            caps = probe()
        with caplog.at_level(logging.WARNING, logger="dlm.hardware.plan"):  # type: ignore[attr-defined]
            plan = resolve(_cfg(precision="bf16"), caps, base_params=1_500_000_000, seq_len=2048)
        assert plan.precision == "bf16"
        assert caplog.records == []  # type: ignore[attr-defined]

    def test_cuda_override_fp32_honored(self) -> None:
        # CUDA default is bf16 (Ampere+) — override to fp32 honored.
        with force_cuda(sm=(8, 0)):
            caps = probe()
        plan = resolve(_cfg(precision="fp32"), caps, base_params=1_500_000_000, seq_len=2048)
        assert plan.precision == "fp32"


class TestAttentionPicker:
    def test_cuda_without_flash_falls_back_to_sdpa(self) -> None:
        with force_cuda(sm=(8, 9)):
            caps = probe()
        # flash_attn probably isn't installed in our dev env; fall back.
        plan = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048)
        if caps.has_flash_attention:
            assert plan.attn_implementation == "flash_attention_2"
        else:
            assert plan.attn_implementation == "sdpa"

    def test_cpu_uses_eager(self) -> None:
        with force_cpu():
            caps = probe()
        plan = resolve(_cfg(), caps, base_params=100_000_000, seq_len=512)
        assert plan.attn_implementation == "eager"

    def test_mps_uses_sdpa(self) -> None:
        with force_mps():
            caps = probe()
        plan = resolve(_cfg(), caps, base_params=135_000_000, seq_len=1024)
        assert plan.attn_implementation == "sdpa"


class TestQloraGating:
    def test_qlora_requested_on_cuda_without_bnb_raises(self) -> None:
        """Refusal matrix (refusals.py) enforces bnb availability before
        the resolver even runs — users can't silently end up with a
        qlora=off plan when they asked for qlora.
        """
        from dlm.hardware.refusals import ResolutionError

        with force_cuda():
            caps = probe()
        if caps.has_bitsandbytes:
            # CUDA dev host with bnb installed: resolver succeeds.
            plan = resolve(_cfg(adapter="qlora"), caps, base_params=1_500_000_000, seq_len=2048)
            assert plan.use_qlora is True
            assert plan.quant_compute_dtype is not None
        else:
            import pytest

            with pytest.raises(ResolutionError, match="bitsandbytes"):
                resolve(
                    _cfg(adapter="qlora"),
                    caps,
                    base_params=1_500_000_000,
                    seq_len=2048,
                )


class TestBatchAndGradAccumResolution:
    def test_auto_micro_batch_and_grad_accum_reach_effective_batch(self) -> None:
        with force_cuda(sm=(8, 9), vram_gb=24.0):
            caps = probe()
        plan = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048)
        assert plan.micro_batch_size >= 1
        assert plan.grad_accum >= 1
        assert plan.effective_batch_size == plan.micro_batch_size * plan.grad_accum
        # Default effective batch is 8; auto-resolver should hit at least 8.
        assert plan.effective_batch_size >= 8

    def test_explicit_micro_batch_respected(self) -> None:
        with force_cuda(sm=(8, 9), vram_gb=24.0):
            caps = probe()
        plan = resolve(_cfg(micro_batch_size=2), caps, base_params=1_500_000_000, seq_len=2048)
        assert plan.micro_batch_size == 2

    def test_explicit_grad_accum_respected(self) -> None:
        with force_cuda(sm=(8, 9), vram_gb=24.0):
            caps = probe()
        plan = resolve(
            _cfg(micro_batch_size=1, grad_accum=16),
            caps,
            base_params=1_500_000_000,
            seq_len=2048,
        )
        assert plan.grad_accum == 16
        assert plan.effective_batch_size == 16

    def test_tight_vram_reduces_micro_batch(self) -> None:
        tight = force_cuda(sm=(8, 9), vram_gb=4.0)
        loose = force_cuda(sm=(8, 9), vram_gb=80.0)
        with tight:
            tight_caps = probe()
        with loose:
            loose_caps = probe()
        tight_plan = resolve(_cfg(), tight_caps, base_params=1_500_000_000, seq_len=2048)
        loose_plan = resolve(_cfg(), loose_caps, base_params=1_500_000_000, seq_len=2048)
        assert tight_plan.micro_batch_size <= loose_plan.micro_batch_size


class TestGradientCheckpointing:
    def test_enabled_when_memory_tight(self) -> None:
        with force_cuda(sm=(8, 9), vram_gb=8.0):
            caps = probe()
        plan = resolve(_cfg(), caps, base_params=3_000_000_000, seq_len=4096)
        assert plan.gradient_checkpointing is True

    def test_disabled_when_plenty_of_headroom(self) -> None:
        with force_cuda(sm=(8, 9), vram_gb=80.0):
            caps = probe()
        plan = resolve(_cfg(), caps, base_params=135_000_000, seq_len=512)
        # Tiny model on huge GPU — no need for grad ckpt.
        assert plan.gradient_checkpointing is False


class TestDpoPhaseAdjustments:
    def test_dpo_halves_micro_batch_relative_to_sft(self) -> None:
        with force_cuda(sm=(8, 9), vram_gb=24.0):
            caps = probe()
        sft = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048, phase="sft")
        dpo = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048, phase="dpo")
        # SFT resolves to ≥2 here; DPO should be strictly less (halved,
        # floor 1). Guard against the degenerate case where SFT = 1.
        if sft.micro_batch_size >= 2:
            assert dpo.micro_batch_size == sft.micro_batch_size // 2
        else:
            assert dpo.micro_batch_size == 1

    def test_dpo_never_drops_below_one(self) -> None:
        # Tight VRAM forces SFT to micro_batch=1; DPO halving must floor
        # at 1, not round to 0.
        with force_cuda(sm=(8, 9), vram_gb=4.0):
            caps = probe()
        dpo = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048, phase="dpo")
        assert dpo.micro_batch_size >= 1

    def test_dpo_peak_vram_exceeds_sft(self) -> None:
        with force_cuda(sm=(8, 9), vram_gb=24.0):
            caps = probe()
        sft = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048, phase="sft")
        dpo = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048, phase="dpo")
        # Even with halved micro-batch, two models resident in VRAM
        # should outweigh the batch reduction on the reference side.
        assert dpo.est_peak_vram_gb > sft.est_peak_vram_gb

    def test_dpo_step_seconds_slower(self) -> None:
        with force_cuda(sm=(8, 9), vram_gb=24.0):
            caps = probe()
        sft = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048, phase="sft")
        dpo = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048, phase="dpo")
        assert dpo.est_step_seconds > sft.est_step_seconds

    def test_dpo_reason_mentions_phase(self) -> None:
        with force_cuda(sm=(8, 9), vram_gb=24.0):
            caps = probe()
        dpo = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048, phase="dpo")
        sft = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048, phase="sft")
        assert "phase=dpo" in dpo.reason
        assert "phase=dpo" not in sft.reason


class TestPlanSerialization:
    def test_to_dict_is_json_friendly(self) -> None:
        import json

        with force_mps():
            caps = probe()
        plan = resolve(_cfg(), caps, base_params=135_000_000, seq_len=512)
        # Round-trip through JSON proves all fields serialize.
        json.dumps(plan.to_dict())

    def test_reason_records_key_choices(self) -> None:
        with force_cuda(sm=(8, 9), vram_gb=24.0):
            caps = probe()
        plan = resolve(_cfg(), caps, base_params=1_500_000_000, seq_len=2048)
        assert "precision=bf16" in plan.reason
        assert "attn=" in plan.reason
