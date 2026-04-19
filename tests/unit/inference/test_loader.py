"""`build_load_kwargs` — config-assembly without touching HF."""

from __future__ import annotations

from dlm.base_models import BASE_MODELS
from dlm.hardware.backend import Backend
from dlm.inference.loader import build_load_kwargs
from dlm.inference.plan import InferencePlan


def _plan(**overrides: object) -> InferencePlan:
    base: dict[str, object] = {
        "backend": Backend.CUDA,
        "precision": "bf16",
        "dequantize_on_load": False,
        "attn_implementation": "sdpa",
        "reason": "test",
    }
    base.update(overrides)
    return InferencePlan(**base)  # type: ignore[arg-type]


class TestBuildLoadKwargs:
    def test_basic_fp16_kwargs(self) -> None:
        spec = BASE_MODELS["smollm2-135m"]
        plan = _plan(backend=Backend.MPS, precision="fp16")
        kwargs = build_load_kwargs(spec, plan, has_bitsandbytes=False)
        assert kwargs["revision"] == spec.revision
        assert kwargs["attn_implementation"] == "sdpa"
        # No quantization config on non-CUDA.
        assert "quantization_config" not in kwargs
        assert "torch_dtype" in kwargs

    def test_dequantize_path_omits_bnb_config(self) -> None:
        """dequantize_on_load=True → no BitsAndBytesConfig even if bnb is installed."""
        spec = BASE_MODELS["smollm2-135m"]
        plan = _plan(dequantize_on_load=True, precision="fp16")
        kwargs = build_load_kwargs(spec, plan, has_bitsandbytes=True)
        assert "quantization_config" not in kwargs

    def test_plain_lora_uses_torch_dtype(self) -> None:
        spec = BASE_MODELS["smollm2-135m"]
        plan = _plan(backend=Backend.CUDA, precision="bf16", dequantize_on_load=False)
        # Has bnb but NO quantization config because this is plain LoRA (the pinned state
        # is checked upstream; the plan encodes the final decision via `dequantize_on_load`
        # + this function's responsibility is only to assemble from the plan).
        # has_bitsandbytes=False → definitely no quantization config.
        kwargs = build_load_kwargs(spec, plan, has_bitsandbytes=False)
        assert "quantization_config" not in kwargs
        assert "torch_dtype" in kwargs
