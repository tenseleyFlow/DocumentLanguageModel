"""`dlm.export.ollama.vl_modelfile.render_vl_modelfile` — rendering + shape.

The VL Modelfile path doesn't exercise end-to-end today (the vendored
llama.cpp can't emit a PaliGemma/InternVL2 GGUF — see
`tests/unit/export/test_arch_probe.py`). These tests cover what will
actually run on the day llama.cpp lands VL support: the render pipeline
from VlModelfileContext → string.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dlm.export.ollama.vl_modelfile import (
    VlModelfileContext,
    render_vl_modelfile,
)


def _fake_vl_spec() -> Any:
    """Minimal spec shape the renderer reads (no full BaseModelSpec needed)."""
    from types import SimpleNamespace

    plan = SimpleNamespace(
        target_size=(224, 224),
        image_token="<image>",
        num_image_tokens=256,
        resize_policy="fixed",
    )
    return SimpleNamespace(
        key="paligemma-3b-mix-224",
        hf_id="google/paligemma-3b-mix-224",
        revision="8d2f7bc9c15d71a00c14f9eb7e4c7b99c79e0a11",
        architecture="PaliGemmaForConditionalGeneration",
        template="paligemma",
        modality="vision-language",
        license_spdx="Gemma",
        license_url="https://ai.google.dev/gemma/terms",
        requires_acceptance=True,
        redistributable=False,
        vl_preprocessor_plan=plan,
        context_length=8192,
        recommended_seq_len=2048,
    )


def _fake_plan() -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(quant="Q4_K_M", merged=False)


@pytest.fixture
def adapter_dir(tmp_path: Path) -> Path:
    """Minimal adapter directory with a tokenizer config.

    The Modelfile render reads `tokenizer_config.json` for EOS+pad
    stops; without it the VL-fallback stops are used alone.
    """
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<eos>"}),
        encoding="utf-8",
    )
    return d


@pytest.fixture
def vl_ctx(adapter_dir: Path) -> VlModelfileContext:
    return VlModelfileContext(
        spec=_fake_vl_spec(),
        plan=_fake_plan(),
        adapter_dir=adapter_dir,
        base_gguf_name="base.Q4_K_M.gguf",
        adapter_gguf_name="adapter.Q4_K_M.gguf",
        dlm_id="01JZZZZZZZZZZZZZZZZZZZZZZZ",
        adapter_version=1,
        dlm_version="v0.10.0",
    )


class TestVlModelfileRender:
    def test_emits_from_and_adapter_directives(self, vl_ctx: VlModelfileContext) -> None:
        out = render_vl_modelfile(vl_ctx)
        assert "FROM ./base.Q4_K_M.gguf" in out
        assert "ADAPTER ./adapter.Q4_K_M.gguf" in out

    def test_template_block_uses_image_directive(self, vl_ctx: VlModelfileContext) -> None:
        """Ollama 0.4+'s `{{ .Image }}` slot is the VL-specific gate."""
        out = render_vl_modelfile(vl_ctx)
        assert "{{ .Image }}" in out
        assert "{{ .Prompt }}" in out
        # System block is conditional.
        assert "{{ if .System }}" in out

    def test_template_uses_triple_quoted_form(self, vl_ctx: VlModelfileContext) -> None:
        """Ollama Modelfile parser wants `TEMPLATE \"\"\"...\"\"\"`.

        The single-line quoted form fails on multi-line templates,
        which a VL shape inherently is.
        """
        out = render_vl_modelfile(vl_ctx)
        assert 'TEMPLATE """' in out

    def test_vl_sampling_defaults(self, vl_ctx: VlModelfileContext) -> None:
        """VL defaults favor determinism over creativity (temp=0.2)."""
        out = render_vl_modelfile(vl_ctx)
        assert "PARAMETER temperature 0.2" in out
        assert "PARAMETER top_p 0.9" in out

    def test_override_temperature_respected(self, vl_ctx: VlModelfileContext) -> None:
        ctx = VlModelfileContext(
            spec=vl_ctx.spec,
            plan=vl_ctx.plan,
            adapter_dir=vl_ctx.adapter_dir,
            base_gguf_name=vl_ctx.base_gguf_name,
            adapter_gguf_name=vl_ctx.adapter_gguf_name,
            dlm_id=vl_ctx.dlm_id,
            adapter_version=vl_ctx.adapter_version,
            override_temperature=0.7,
        )
        out = render_vl_modelfile(ctx)
        assert "PARAMETER temperature 0.7" in out

    def test_adapter_eos_merged_into_stops(self, vl_ctx: VlModelfileContext) -> None:
        """The `<eos>` token from tokenizer_config.json is a PARAMETER stop."""
        out = render_vl_modelfile(vl_ctx)
        # JSON-quoted `"<eos>"` appears in the stop line.
        assert 'PARAMETER stop "<eos>"' in out

    def test_merged_path_omits_adapter_directive(self, vl_ctx: VlModelfileContext) -> None:
        merged_ctx = VlModelfileContext(
            spec=vl_ctx.spec,
            plan=vl_ctx.plan,
            adapter_dir=vl_ctx.adapter_dir,
            base_gguf_name="merged.Q4_K_M.gguf",
            adapter_gguf_name=None,
            dlm_id=vl_ctx.dlm_id,
            adapter_version=vl_ctx.adapter_version,
        )
        out = render_vl_modelfile(merged_ctx)
        assert "FROM ./merged.Q4_K_M.gguf" in out
        assert "ADAPTER " not in out

    def test_system_prompt_emitted_as_directive(self, vl_ctx: VlModelfileContext) -> None:
        ctx = VlModelfileContext(
            spec=vl_ctx.spec,
            plan=vl_ctx.plan,
            adapter_dir=vl_ctx.adapter_dir,
            base_gguf_name=vl_ctx.base_gguf_name,
            adapter_gguf_name=vl_ctx.adapter_gguf_name,
            dlm_id=vl_ctx.dlm_id,
            adapter_version=vl_ctx.adapter_version,
            system_prompt="Describe the image concisely.",
        )
        out = render_vl_modelfile(ctx)
        assert 'SYSTEM "Describe the image concisely."' in out

    def test_ends_with_newline(self, vl_ctx: VlModelfileContext) -> None:
        """Every text file; confirms no trailing-junk regressions."""
        out = render_vl_modelfile(vl_ctx)
        assert out.endswith("\n")


class TestStopsFallback:
    def test_paligemma_uses_gemma_style_stop(self, tmp_path: Path) -> None:
        """PaliGemma's fallback is Gemma's `<eos>` alone — not im_end."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()  # no tokenizer_config.json inside
        ctx = VlModelfileContext(
            spec=_fake_vl_spec(),  # PaliGemmaForConditionalGeneration
            plan=_fake_plan(),
            adapter_dir=adapter_dir,
            base_gguf_name="base.Q4_K_M.gguf",
            adapter_gguf_name=None,
            dlm_id="01JZZZZZZZZZZZZZZZZZZZZZZZ",
            adapter_version=1,
        )
        out = render_vl_modelfile(ctx)
        assert 'PARAMETER stop "<eos>"' in out
        # chatml-style stops must NOT appear on a PaliGemma Modelfile —
        # otherwise Ollama would pre-emptively stop on a token the
        # Gemma tokenizer never emits.
        assert 'PARAMETER stop "<|im_end|>"' not in out

    def test_qwen2_vl_uses_chatml_stops(self, tmp_path: Path) -> None:
        """Qwen2-VL uses chatml: `<|im_end|>` + `<|endoftext|>`."""
        spec = _fake_vl_spec()
        object.__setattr__(spec, "architecture", "Qwen2VLForConditionalGeneration")
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        ctx = VlModelfileContext(
            spec=spec,
            plan=_fake_plan(),
            adapter_dir=adapter_dir,
            base_gguf_name="base.Q4_K_M.gguf",
            adapter_gguf_name=None,
            dlm_id="01JZZZZZZZZZZZZZZZZZZZZZZZ",
            adapter_version=1,
        )
        out = render_vl_modelfile(ctx)
        assert 'PARAMETER stop "<|im_end|>"' in out
        assert 'PARAMETER stop "<|endoftext|>"' in out
        assert 'PARAMETER stop "<eos>"' not in out

    def test_unknown_arch_falls_back_to_union(self, tmp_path: Path) -> None:
        """Unknown architectures fall back to the safe union."""
        spec = _fake_vl_spec()
        object.__setattr__(spec, "architecture", "SomeNewVLModel")
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        ctx = VlModelfileContext(
            spec=spec,
            plan=_fake_plan(),
            adapter_dir=adapter_dir,
            base_gguf_name="base.Q4_K_M.gguf",
            adapter_gguf_name=None,
            dlm_id="01JZZZZZZZZZZZZZZZZZZZZZZZ",
            adapter_version=1,
        )
        out = render_vl_modelfile(ctx)
        assert 'PARAMETER stop "<|im_end|>"' in out
        assert 'PARAMETER stop "<eos>"' in out

    def test_mistral3_uses_mistral_family_stops(self, tmp_path: Path) -> None:
        spec = _fake_vl_spec()
        object.__setattr__(spec, "architecture", "Mistral3ForConditionalGeneration")
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        ctx = VlModelfileContext(
            spec=spec,
            plan=_fake_plan(),
            adapter_dir=adapter_dir,
            base_gguf_name="base.Q4_K_M.gguf",
            adapter_gguf_name=None,
            dlm_id="01JZZZZZZZZZZZZZZZZZZZZZZZ",
            adapter_version=1,
        )
        out = render_vl_modelfile(ctx)
        assert 'PARAMETER stop "</s>"' in out
        assert 'PARAMETER stop "[INST]"' in out
