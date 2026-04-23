"""Sprint 40 closeout checks for the Mixtral template row."""

from __future__ import annotations

import json
from pathlib import Path

from dlm.base_models import BASE_MODELS
from dlm.export.ollama.modelfile import ModelfileContext, render_modelfile
from dlm.export.plan import ExportPlan


def _adapter_dir(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "</s>", "added_tokens_decoder": {}}),
        encoding="utf-8",
    )
    return adapter


def test_mixtral_registry_row_renders_through_mistral_template(tmp_path: Path) -> None:
    spec = BASE_MODELS["mixtral-8x7b-instruct"]
    text = render_modelfile(
        ModelfileContext(
            spec=spec,
            plan=ExportPlan(quant="Q4_K_M", merged=False),
            adapter_dir=_adapter_dir(tmp_path),
            base_gguf_name="base.gguf",
            adapter_gguf_name="adapter.gguf",
            dlm_id="01TEST",
            adapter_version=1,
        )
    )
    assert spec.modality == "text-moe"
    assert "[INST]" in text
    assert 'PARAMETER stop "[INST]"' in text
