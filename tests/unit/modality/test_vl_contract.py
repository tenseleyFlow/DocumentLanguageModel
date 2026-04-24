"""Direct coverage for VL runtime contract guardrails."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dlm.modality.errors import ProcessorContractError
from dlm.modality.vl_contract import ensure_supported_vl_runtime, validate_loaded_vl_processor


def test_ensure_supported_vl_runtime_is_noop_for_non_vl_specs() -> None:
    ensure_supported_vl_runtime(
        SimpleNamespace(modality="text", architecture="Anything", key="demo")
    )


def test_ensure_supported_vl_runtime_is_noop_for_supported_vl_architecture() -> None:
    ensure_supported_vl_runtime(
        SimpleNamespace(
            modality="vision-language", architecture="Qwen2VLForConditionalGeneration", key="demo"
        )
    )


def test_ensure_supported_vl_runtime_rejects_internvl_family() -> None:
    with pytest.raises(ProcessorContractError, match="InternVL-family VL model"):
        ensure_supported_vl_runtime(
            SimpleNamespace(
                modality="vision-language", architecture="InternVLChatModel", key="internvl"
            )
        )


def test_validate_loaded_vl_processor_is_noop_for_non_vl_specs() -> None:
    processor = object()
    assert (
        validate_loaded_vl_processor(
            SimpleNamespace(modality="text", architecture="Demo", key="demo"), processor
        )
        is processor
    )


def test_validate_loaded_vl_processor_accepts_processor_with_image_processor() -> None:
    processor = SimpleNamespace(image_processor=object())
    assert (
        validate_loaded_vl_processor(
            SimpleNamespace(modality="vision-language", architecture="Demo", key="demo"),
            processor,
        )
        is processor
    )


def test_validate_loaded_vl_processor_delegates_internvl_refusal() -> None:
    with pytest.raises(ProcessorContractError, match="InternVL-family VL model"):
        validate_loaded_vl_processor(
            SimpleNamespace(
                modality="vision-language", architecture="InternVLChatModel", key="internvl"
            ),
            SimpleNamespace(),
        )


def test_validate_loaded_vl_processor_rejects_missing_image_processor() -> None:
    with pytest.raises(ProcessorContractError, match="without an `image_processor` attribute"):
        validate_loaded_vl_processor(
            SimpleNamespace(modality="vision-language", architecture="Demo", key="demo"),
            SimpleNamespace(),
        )
