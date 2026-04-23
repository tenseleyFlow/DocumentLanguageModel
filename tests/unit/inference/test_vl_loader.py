"""`load_for_vl_inference` guardrails for remote-code VL families."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dlm.base_models.schema import BaseModelSpec, VlPreprocessorPlan
from dlm.inference.vl_loader import load_for_vl_inference
from dlm.modality import ProcessorContractError


def _internvl_spec() -> BaseModelSpec:
    return BaseModelSpec(
        key="internvl-test",
        hf_id="test/internvl",
        revision="a" * 40,
        architecture="InternVLChatModel",
        params=2_200_000_000,
        target_modules=["q_proj"],
        template="internvl2",
        gguf_arch="internvl2",
        tokenizer_pre="internvl2",
        license_spdx="Apache-2.0",
        redistributable=True,
        trust_remote_code=True,
        size_gb_fp16=4.4,
        context_length=8192,
        recommended_seq_len=2048,
        modality="vision-language",
        vl_preprocessor_plan=VlPreprocessorPlan(
            target_size=(448, 448),
            resize_policy="dynamic",
            image_token="<image>",
            num_image_tokens=256,
        ),
    )


class TestLoadForVlInference:
    def test_internvl_family_refused_before_model_load(self) -> None:
        with pytest.raises(ProcessorContractError, match="InternVL-family VL model"):
            load_for_vl_inference(
                store=MagicMock(),
                spec=_internvl_spec(),
                caps=MagicMock(),
                adapter_name=None,
            )
