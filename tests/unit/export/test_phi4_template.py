"""Sprint 40 closeout checks for the Phi-4 reasoning template row."""

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
        json.dumps({"eos_token": "<|end|>", "added_tokens_decoder": {}}),
        encoding="utf-8",
    )
    return adapter


def test_phi4_reasoning_template_keeps_phi_system_preamble_and_stops(tmp_path: Path) -> None:
    text = render_modelfile(
        ModelfileContext(
            spec=BASE_MODELS["phi-4-mini-reasoning"],
            plan=ExportPlan(quant="Q4_K_M", merged=False),
            adapter_dir=_adapter_dir(tmp_path),
            base_gguf_name="base.gguf",
            adapter_gguf_name="adapter.gguf",
            dlm_id="01TEST",
            adapter_version=1,
        )
    )
    assert "Your name is Phi, an AI math expert developed by Microsoft." in text
    assert 'PARAMETER stop "<|assistant|>"' in text
    assert "PARAMETER temperature 0.6" in text
