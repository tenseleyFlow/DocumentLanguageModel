"""Direct coverage for modality dispatch wrapper modules."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dlm.base_models import BaseModelSpec
from dlm.modality.audio import AudioLanguageModality
from dlm.modality.errors import UnknownModalityError
from dlm.modality.registry import TextModality, _unknown, modality_for
from dlm.modality.text import TextModality as ReexportedTextModality
from dlm.modality.vl import VisionLanguageModality


def _minimal_text_spec(*, modality: str = "text") -> BaseModelSpec:
    return BaseModelSpec.model_validate(
        {
            "key": "demo-1b",
            "hf_id": "org/demo-1b",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "architecture": "DemoForCausalLM",
            "params": 1_000_000_000,
            "target_modules": ["q_proj", "v_proj"],
            "template": "chatml",
            "gguf_arch": "demo",
            "tokenizer_pre": "demo",
            "license_spdx": "Apache-2.0",
            "license_url": None,
            "requires_acceptance": False,
            "redistributable": True,
            "size_gb_fp16": 2.0,
            "context_length": 4096,
            "recommended_seq_len": 2048,
            "modality": modality,
        }
    )


def test_text_module_reexports_text_modality() -> None:
    assert ReexportedTextModality is TextModality


def test_text_dispatch_defaults_are_noops() -> None:
    dispatch = TextModality()

    assert dispatch.load_processor(_minimal_text_spec()) is None
    assert (
        dispatch.dispatch_export(
            store=object(),
            spec=_minimal_text_spec(),
            adapter_name=None,
            quant=None,
            merged=False,
            adapter_mix_raw=None,
        )
        is None
    )


def test_unknown_error_contains_registration_hint() -> None:
    err = _unknown("mystery")
    assert isinstance(err, UnknownModalityError)
    assert "Register a ModalityDispatch subclass" in str(err)


def test_modality_for_unknown_modality_raises() -> None:
    with pytest.raises(UnknownModalityError, match="mystery"):
        modality_for(SimpleNamespace(modality="mystery"))


def test_audio_modality_loads_processor_and_dispatches_export() -> None:
    dispatch = AudioLanguageModality()
    spec = SimpleNamespace()

    with (
        patch("dlm.train.loader.load_processor", return_value="processor") as load_processor,
        patch("dlm.export.dispatch.dispatch_audio_export", return_value="audio-export") as export,
    ):
        processor = dispatch.load_processor(spec)
        result = dispatch.dispatch_export(
            store="store",
            spec=spec,
            adapter_name="adapter",
            quant="q4_k_m",
            merged=False,
            adapter_mix_raw="named",
        )

    assert processor == "processor"
    load_processor.assert_called_once_with(spec)
    assert result == "audio-export"
    export.assert_called_once_with(
        store="store",
        spec=spec,
        adapter_name="adapter",
        quant="q4_k_m",
        merged=False,
        adapter_mix_raw="named",
    )


def test_vl_modality_loads_processor_and_dispatches_export() -> None:
    dispatch = VisionLanguageModality()
    spec = SimpleNamespace()
    context = {"emit": "gguf"}

    with (
        patch("dlm.train.loader.load_processor", return_value="processor") as load_processor,
        patch("dlm.export.dispatch.dispatch_vl_export", return_value="vl-export") as export,
    ):
        processor = dispatch.load_processor(spec)
        result = dispatch.dispatch_export(
            store="store",
            spec=spec,
            adapter_name="adapter",
            quant="q8_0",
            merged=True,
            adapter_mix_raw=None,
            gguf_emission_context=context,
        )

    assert processor == "processor"
    load_processor.assert_called_once_with(spec)
    assert result == "vl-export"
    export.assert_called_once_with(
        store="store",
        spec=spec,
        adapter_name="adapter",
        quant="q8_0",
        merged=True,
        adapter_mix_raw=None,
        gguf_emission_context=context,
    )
