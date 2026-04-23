"""Sprint 40 closeout checks for the Qwen3 reasoning-template row."""

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
        json.dumps({"eos_token": "<|im_end|>", "added_tokens_decoder": {}}),
        encoding="utf-8",
    )
    return adapter


def test_qwen3_thinking_row_uses_distinct_reasoning_template_defaults(tmp_path: Path) -> None:
    text = render_modelfile(
        ModelfileContext(
            spec=BASE_MODELS["qwen3-1.7b-thinking"],
            plan=ExportPlan(quant="Q4_K_M", merged=False),
            adapter_dir=_adapter_dir(tmp_path),
            base_gguf_name="base.gguf",
            adapter_gguf_name="adapter.gguf",
            dlm_id="01TEST",
            adapter_version=1,
        )
    )
    assert "PARAMETER temperature 0.6" in text
    assert "PARAMETER top_p 0.95" in text
    assert "<|im_start|>assistant" in text
