"""InferencePlan resolver — audit F05 cross-hardware coverage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from dlm.hardware.backend import Backend
from dlm.inference.plan import InferencePlan, resolve_inference


def _caps(
    *,
    backend: Backend,
    supports_bf16: bool = False,
    has_bitsandbytes: bool = False,
    has_flash_attention: bool = False,
) -> object:
    return SimpleNamespace(
        backend=backend,
        supports_bf16=supports_bf16,
        has_bitsandbytes=has_bitsandbytes,
        has_flash_attention=has_flash_attention,
    )


def _write_pinned(adapter_dir: Path, *, bnb: str | None) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "pinned_versions.json").write_text(
        json.dumps({"torch": "2.4.0", "bitsandbytes": bnb})
    )


def _write_training_run(adapter_dir: Path, *, use_qlora: bool) -> None:
    """Write `training_run.json` — the audit-05 M1 authoritative QLoRA signal."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "training_run.json").write_text(json.dumps({"use_qlora": use_qlora}))


class TestQLoRAOnCUDAWithBnb:
    def test_loads_4bit_native(self, tmp_path: Path) -> None:
        _write_pinned(tmp_path, bnb="0.43.1")
        plan = resolve_inference(
            tmp_path, _caps(backend=Backend.CUDA, supports_bf16=True, has_bitsandbytes=True)
        )
        assert plan.backend == Backend.CUDA
        assert plan.precision == "bf16"
        assert plan.dequantize_on_load is False
        assert "4-bit" in plan.reason


class TestQLoRAOnCUDAWithoutBnb:
    def test_dequantizes(self, tmp_path: Path) -> None:
        _write_pinned(tmp_path, bnb="0.43.1")
        plan = resolve_inference(tmp_path, _caps(backend=Backend.CUDA, has_bitsandbytes=False))
        assert plan.dequantize_on_load is True
        assert plan.precision == "fp16"
        assert "bitsandbytes not installed" in plan.reason


class TestQLoRAOnMPS:
    """Audit F05 canonical case — CUDA-trained QLoRA resumed on Apple Silicon."""

    def test_dequantizes_to_fp16(self, tmp_path: Path) -> None:
        _write_pinned(tmp_path, bnb="0.43.1")
        plan = resolve_inference(tmp_path, _caps(backend=Backend.MPS))
        assert plan.backend == Backend.MPS
        assert plan.precision == "fp16"
        assert plan.dequantize_on_load is True
        assert plan.attn_implementation == "sdpa"
        assert "QLoRA adapter on mps host" in plan.reason


class TestLoRANonCUDA:
    def test_mps_plain_lora(self, tmp_path: Path) -> None:
        _write_pinned(tmp_path, bnb=None)
        plan = resolve_inference(tmp_path, _caps(backend=Backend.MPS))
        assert plan.precision == "fp16"
        assert plan.dequantize_on_load is False

    def test_cpu_plain_lora(self, tmp_path: Path) -> None:
        _write_pinned(tmp_path, bnb=None)
        plan = resolve_inference(tmp_path, _caps(backend=Backend.CPU))
        assert plan.dequantize_on_load is False


class TestLoRAOnCUDA:
    def test_bf16_when_supported(self, tmp_path: Path) -> None:
        _write_pinned(tmp_path, bnb=None)
        plan = resolve_inference(tmp_path, _caps(backend=Backend.CUDA, supports_bf16=True))
        assert plan.precision == "bf16"
        assert plan.dequantize_on_load is False

    def test_fp16_when_bf16_unsupported(self, tmp_path: Path) -> None:
        _write_pinned(tmp_path, bnb=None)
        plan = resolve_inference(tmp_path, _caps(backend=Backend.CUDA, supports_bf16=False))
        assert plan.precision == "fp16"


class TestAttnImplPick:
    def test_flash_attn_when_available(self, tmp_path: Path) -> None:
        _write_pinned(tmp_path, bnb="0.43.1")
        plan = resolve_inference(
            tmp_path,
            _caps(
                backend=Backend.CUDA,
                supports_bf16=True,
                has_bitsandbytes=True,
                has_flash_attention=True,
            ),
        )
        assert plan.attn_implementation == "flash_attention_2"

    def test_sdpa_default(self, tmp_path: Path) -> None:
        _write_pinned(tmp_path, bnb=None)
        plan = resolve_inference(tmp_path, _caps(backend=Backend.CUDA))
        assert plan.attn_implementation == "sdpa"


class TestAuditM1UseQloraFlag:
    """Audit-05 M1: `training_run.json` is the authoritative QLoRA signal.

    Before the M1 fix, any non-None `bitsandbytes` pin triggered the
    dequantize path — false-positive for plain LoRA runs on CUDA+bnb
    hosts. These tests verify the explicit flag wins over the pin.
    """

    def test_plain_lora_with_bnb_installed_no_dequantize(self, tmp_path: Path) -> None:
        """CUDA+bnb box trained plain LoRA → moving to MPS must NOT dequantize."""
        # bnb pinned (because it was installed on the training host)
        # but use_qlora=False (because the user chose plain LoRA).
        _write_pinned(tmp_path, bnb="0.43.1")
        _write_training_run(tmp_path, use_qlora=False)
        plan = resolve_inference(tmp_path, _caps(backend=Backend.MPS))
        assert plan.dequantize_on_load is False
        assert "QLoRA adapter" not in plan.reason

    def test_qlora_flag_wins_over_missing_bnb_pin(self, tmp_path: Path) -> None:
        """Explicit use_qlora=True triggers dequantize even without bnb pin."""
        _write_pinned(tmp_path, bnb=None)
        _write_training_run(tmp_path, use_qlora=True)
        plan = resolve_inference(tmp_path, _caps(backend=Backend.MPS))
        assert plan.dequantize_on_load is True
        assert "QLoRA adapter on mps host" in plan.reason

    def test_legacy_fallback_when_training_run_missing(self, tmp_path: Path) -> None:
        """Pre-audit-05 adapters only have `pinned_versions.json`.

        Bnb pin present → legacy inference path treats as QLoRA.
        """
        _write_pinned(tmp_path, bnb="0.43.1")
        # NO training_run.json → fall back.
        plan = resolve_inference(tmp_path, _caps(backend=Backend.MPS))
        assert plan.dequantize_on_load is True

    def test_malformed_training_run_falls_back(self, tmp_path: Path) -> None:
        (tmp_path / "training_run.json").write_text("not json {{{")
        _write_pinned(tmp_path, bnb="0.43.1")
        plan = resolve_inference(tmp_path, _caps(backend=Backend.MPS))
        # Falls back to the pin heuristic.
        assert plan.dequantize_on_load is True


class TestMissingPinnedFile:
    def test_no_pinned_versions_treated_as_lora(self, tmp_path: Path) -> None:
        """Missing `pinned_versions.json` is conservative: assume LoRA."""
        plan = resolve_inference(tmp_path, _caps(backend=Backend.MPS))
        assert plan.dequantize_on_load is False

    def test_malformed_pinned_file_treated_as_lora(self, tmp_path: Path) -> None:
        (tmp_path / "pinned_versions.json").write_text("not json {{{")
        plan = resolve_inference(tmp_path, _caps(backend=Backend.MPS))
        assert plan.dequantize_on_load is False


class TestPlanSerialization:
    def test_to_dict_is_json_friendly(self, tmp_path: Path) -> None:
        _write_pinned(tmp_path, bnb=None)
        plan = resolve_inference(tmp_path, _caps(backend=Backend.MPS))
        data = plan.to_dict()
        # Round-trip via json to prove serializability.
        encoded = json.dumps(data)
        decoded = json.loads(encoded)
        assert decoded["backend"] == "mps"
        assert decoded["precision"] == "fp16"

    def test_plan_is_frozen(self, tmp_path: Path) -> None:
        import dataclasses

        _write_pinned(tmp_path, bnb=None)
        plan = resolve_inference(tmp_path, _caps(backend=Backend.MPS))
        assert isinstance(plan, InferencePlan)
        try:
            plan.precision = "bf16"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError("frozen=True not enforced")
