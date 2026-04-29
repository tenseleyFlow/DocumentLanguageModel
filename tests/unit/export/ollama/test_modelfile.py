"""Modelfile shape + SYSTEM injection defense."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlm.base_models import BASE_MODELS
from dlm.export.ollama.errors import ModelfileError
from dlm.export.ollama.modelfile import ModelfileContext, render_modelfile
from dlm.export.plan import ExportPlan

_SPEC = BASE_MODELS["smollm2-135m"]


def _adapter_dir(tmp_path: Path, **extra: object) -> Path:
    """Write a minimal adapter dir with a tokenizer_config.json."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    cfg = {
        "eos_token": "<|im_end|>",
        "added_tokens_decoder": {
            "50265": {"content": "<|pad|>", "special": True},
        },
        **extra,
    }
    (adapter / "tokenizer_config.json").write_text(json.dumps(cfg))
    return adapter


def _ctx(
    tmp_path: Path,
    *,
    spec: object | None = None,
    merged: bool = False,
    system_prompt: str | None = None,
    adapter: Path | None = None,
    training_sequence_len: int | None = None,
    override_temperature: float | None = None,
    override_top_p: float | None = None,
    draft_model_ollama_name: str | None = None,
) -> ModelfileContext:
    plan = ExportPlan(quant="Q4_K_M", merged=merged)
    return ModelfileContext(
        spec=_SPEC if spec is None else spec,
        plan=plan,
        adapter_dir=adapter or _adapter_dir(tmp_path),
        base_gguf_name="base.Q4_K_M.gguf",
        adapter_gguf_name=None if merged else "adapter.gguf",
        dlm_id="01TEST",
        adapter_version=7,
        system_prompt=system_prompt,
        training_sequence_len=training_sequence_len,
        override_temperature=override_temperature,
        override_top_p=override_top_p,
        draft_model_ollama_name=draft_model_ollama_name,
    )


class TestShape:
    def test_header_metadata(self, tmp_path: Path) -> None:
        text = render_modelfile(_ctx(tmp_path))
        assert "# dlm_id: 01TEST" in text
        assert "# adapter_version: 7" in text
        assert f"# base_model: {_SPEC.key}" in text
        assert "# quant: Q4_K_M" in text

    def test_from_and_adapter_lines(self, tmp_path: Path) -> None:
        text = render_modelfile(_ctx(tmp_path))
        assert "FROM ./base.Q4_K_M.gguf" in text
        assert "ADAPTER ./adapter.gguf" in text

    def test_merged_has_no_adapter_line(self, tmp_path: Path) -> None:
        text = render_modelfile(_ctx(tmp_path, merged=True))
        assert "FROM ./base.Q4_K_M.gguf" in text
        assert "ADAPTER" not in text

    def test_template_block_present(self, tmp_path: Path) -> None:
        text = render_modelfile(_ctx(tmp_path))
        assert 'TEMPLATE """' in text
        assert "<|im_start|>" in text  # chatml dialect

    def test_no_template_when_include_template_false(self, tmp_path: Path) -> None:
        """Audit 13 M13.1: ``--no-template`` must actually suppress the
        TEMPLATE block in the emitted Modelfile, not just the preflight."""
        ctx = _ctx(tmp_path)
        # Bypass the frozen-dataclass replace dance: rebuild with a plan
        # that has include_template=False.
        plan = ExportPlan(quant="Q4_K_M", merged=False, include_template=False)
        ctx_no_tmpl = ModelfileContext(
            spec=ctx.spec,
            plan=plan,
            adapter_dir=ctx.adapter_dir,
            base_gguf_name=ctx.base_gguf_name,
            adapter_gguf_name=ctx.adapter_gguf_name,
            dlm_id=ctx.dlm_id,
            adapter_version=ctx.adapter_version,
            system_prompt=ctx.system_prompt,
            training_sequence_len=ctx.training_sequence_len,
            override_temperature=ctx.override_temperature,
            override_top_p=ctx.override_top_p,
            draft_model_ollama_name=ctx.draft_model_ollama_name,
        )
        text = render_modelfile(ctx_no_tmpl)
        assert "TEMPLATE" not in text
        # Other directives still emit normally.
        assert "FROM ./base.Q4_K_M.gguf" in text
        assert "PARAMETER temperature" in text

    def test_params_emitted(self, tmp_path: Path) -> None:
        text = render_modelfile(_ctx(tmp_path))
        assert "PARAMETER temperature" in text
        assert "PARAMETER top_p" in text

    def test_num_ctx_omitted_when_unset(self, tmp_path: Path) -> None:
        """No `training_sequence_len` → no `PARAMETER num_ctx` (Ollama default)."""
        text = render_modelfile(_ctx(tmp_path))
        assert "PARAMETER num_ctx" not in text

    def test_num_ctx_inherited_from_training_seq_len(self, tmp_path: Path) -> None:
        """Audit-04 Q1: frontmatter sequence_len flows into the Modelfile."""
        text = render_modelfile(_ctx(tmp_path, training_sequence_len=8192))
        assert "PARAMETER num_ctx 8192" in text

    def test_num_ctx_capped_at_spec_context_length(self, tmp_path: Path) -> None:
        """A user writing `sequence_len: 99999` can't exceed the base's positional table."""
        text = render_modelfile(_ctx(tmp_path, training_sequence_len=99_999))
        # smollm2-135m tops out at 8192.
        assert f"PARAMETER num_ctx {_SPEC.context_length}" in text
        assert "PARAMETER num_ctx 99999" not in text

    def test_override_temperature(self, tmp_path: Path) -> None:
        """Audit-04 Q5: frontmatter export.default_temperature overrides the dialect default."""
        text = render_modelfile(_ctx(tmp_path, override_temperature=0.2))
        assert "PARAMETER temperature 0.2" in text

    def test_draft_model_omitted_by_default(self, tmp_path: Path) -> None:
        """Sprint 12.5: no draft set → no `PARAMETER draft_model`."""
        text = render_modelfile(_ctx(tmp_path))
        assert "PARAMETER draft_model" not in text

    def test_draft_model_emitted_when_set(self, tmp_path: Path) -> None:
        """Sprint 12.5: `PARAMETER draft_model <tag>` appears + pull reminder."""
        text = render_modelfile(_ctx(tmp_path, draft_model_ollama_name="qwen2.5:0.5b"))
        assert "PARAMETER draft_model qwen2.5:0.5b" in text
        assert "ollama pull qwen2.5:0.5b" in text

    def test_override_top_p(self, tmp_path: Path) -> None:
        """Audit-04 Q5: frontmatter export.default_top_p overrides the dialect default."""
        text = render_modelfile(_ctx(tmp_path, override_top_p=0.5))
        assert "PARAMETER top_p 0.5" in text

    def test_unset_overrides_fall_back_to_dialect(self, tmp_path: Path) -> None:
        """When overrides are None, the spec + dialect defaults are emitted."""
        text = render_modelfile(_ctx(tmp_path))
        # chatml defaults aren't zero/one, but we can assert both PARAMETER lines exist.
        assert "PARAMETER temperature 0.2" not in text  # would be present if override bled in

    def test_license_line_present(self, tmp_path: Path) -> None:
        text = render_modelfile(_ctx(tmp_path))
        assert 'LICENSE "Apache-2.0"' in text

    def test_license_line_omitted_when_spec_has_no_spdx(self, tmp_path: Path) -> None:
        spec = _SPEC.model_copy(update={"license_spdx": ""})
        text = render_modelfile(_ctx(tmp_path, spec=spec))
        assert "LICENSE " not in text

    def test_trailing_newline(self, tmp_path: Path) -> None:
        assert render_modelfile(_ctx(tmp_path)).endswith("\n")

    def test_reasoning_tuned_spec_drives_default_temperature(self, tmp_path: Path) -> None:
        text = render_modelfile(_ctx(tmp_path, spec=BASE_MODELS["qwen3-1.7b"]))
        assert "PARAMETER temperature 0.6" in text


class TestStops:
    def test_dialect_defaults_emitted(self, tmp_path: Path) -> None:
        text = render_modelfile(_ctx(tmp_path))
        assert 'PARAMETER stop "<|im_end|>"' in text
        assert 'PARAMETER stop "<|endoftext|>"' in text

    def test_adapter_added_token_becomes_stop(self, tmp_path: Path) -> None:
        """Audit F06: added-by-training special tokens feed the stop list."""
        text = render_modelfile(_ctx(tmp_path))
        assert 'PARAMETER stop "<|pad|>"' in text

    def test_dedup_between_dialect_and_adapter(self, tmp_path: Path) -> None:
        """`<|im_end|>` from both dialect defaults + adapter eos appears once."""
        text = render_modelfile(_ctx(tmp_path))
        count = text.count('PARAMETER stop "<|im_end|>"')
        assert count == 1

    def test_missing_tokenizer_config_falls_back_to_dialect(self, tmp_path: Path) -> None:
        adapter = tmp_path / "bare"
        adapter.mkdir()
        text = render_modelfile(_ctx(tmp_path, adapter=adapter))
        assert 'PARAMETER stop "<|im_end|>"' in text

    def test_malformed_tokenizer_config_raises(self, tmp_path: Path) -> None:
        adapter = tmp_path / "broken"
        adapter.mkdir()
        (adapter / "tokenizer_config.json").write_text("{not json")
        with pytest.raises(ModelfileError):
            render_modelfile(_ctx(tmp_path, adapter=adapter))

    def test_eos_dict_content_read(self, tmp_path: Path) -> None:
        adapter = tmp_path / "dict-eos"
        adapter.mkdir()
        (adapter / "tokenizer_config.json").write_text(
            json.dumps({"eos_token": {"content": "<|my_eos|>"}})
        )
        text = render_modelfile(_ctx(tmp_path, adapter=adapter))
        assert 'PARAMETER stop "<|my_eos|>"' in text


class TestSystemInjection:
    def test_no_prompt_no_line(self, tmp_path: Path) -> None:
        text = render_modelfile(_ctx(tmp_path))
        assert "SYSTEM" not in text

    def test_empty_prompt_no_line(self, tmp_path: Path) -> None:
        text = render_modelfile(_ctx(tmp_path, system_prompt="   \n  "))
        assert "SYSTEM" not in text

    def test_plain_prompt_escaped(self, tmp_path: Path) -> None:
        text = render_modelfile(_ctx(tmp_path, system_prompt="be terse"))
        assert 'SYSTEM "be terse"' in text

    def test_quote_in_prompt_escaped(self, tmp_path: Path) -> None:
        """`"` in the prompt must be escaped, not close the SYSTEM string."""
        malicious = 'close"\nPARAMETER foo bar'
        text = render_modelfile(_ctx(tmp_path, system_prompt=malicious))
        # json.dumps escapes `"` to `\"` and `\n` to `\\n`.
        assert 'SYSTEM "close\\"' in text
        # The injected directive must not be at column 0 as its own line.
        for line in text.splitlines():
            assert not line.startswith("PARAMETER foo")

    def test_newline_in_prompt_escaped(self, tmp_path: Path) -> None:
        text = render_modelfile(_ctx(tmp_path, system_prompt="line1\nline2"))
        # Newline encoded as `\n` inside the quoted string — never a raw newline.
        system_lines = [line for line in text.splitlines() if line.startswith("SYSTEM ")]
        assert len(system_lines) == 1
        assert "\\n" in system_lines[0]
