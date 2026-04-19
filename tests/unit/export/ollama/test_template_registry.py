"""Dialect → Go template registry."""

from __future__ import annotations

import pytest

from dlm.export.ollama.errors import TemplateRegistryError
from dlm.export.ollama.template_registry import (
    get_template,
    load_template_text,
    registered_dialects,
)


class TestRegistryCoverage:
    def test_all_four_dialects_registered(self) -> None:
        assert set(registered_dialects()) == {"chatml", "llama3", "phi3", "mistral"}

    @pytest.mark.parametrize("dialect", ["chatml", "llama3", "phi3", "mistral"])
    def test_each_template_file_exists(self, dialect: str) -> None:
        row = get_template(dialect)
        assert row.template_path.is_file()
        body = row.read_template()
        assert body  # non-empty

    @pytest.mark.parametrize("dialect", ["chatml", "llama3", "phi3", "mistral"])
    def test_each_has_default_stops(self, dialect: str) -> None:
        row = get_template(dialect)
        assert row.default_stops  # at least one

    @pytest.mark.parametrize(
        ("dialect", "required"),
        [
            ("chatml", {"<|im_end|>", "<|im_start|>"}),
            ("llama3", {"<|eot_id|>", "<|start_header_id|>"}),
            ("phi3", {"<|end|>", "<|user|>", "<|assistant|>"}),
            ("mistral", {"</s>", "[/INST]", "[INST]"}),
        ],
    )
    def test_role_delimiters_are_stops(self, dialect: str, required: set[str]) -> None:
        """Audit-04 Q4: role-delimiter tokens in each template must appear as stops
        so the model can't synthesize a new turn instead of yielding."""
        row = get_template(dialect)
        stops = set(row.default_stops)
        missing = required - stops
        assert not missing, f"{dialect} missing role-delimiter stops: {missing}"


class TestRegistryLookup:
    def test_unknown_dialect_raises(self) -> None:
        with pytest.raises(TemplateRegistryError, match="unknown template dialect"):
            get_template("claude")

    def test_load_template_text_is_file_bytes(self) -> None:
        text = load_template_text("chatml")
        assert "{{" in text  # Go template syntax present


class TestDialectShapes:
    """Lightweight sanity: required Go markers appear in each template."""

    def test_chatml_has_im_markers(self) -> None:
        text = load_template_text("chatml")
        assert "<|im_start|>" in text
        assert "<|im_end|>" in text

    def test_llama3_has_header_markers(self) -> None:
        text = load_template_text("llama3")
        assert "<|start_header_id|>" in text
        assert "<|eot_id|>" in text

    def test_phi3_has_end_marker(self) -> None:
        text = load_template_text("phi3")
        assert "<|end|>" in text

    def test_mistral_has_inst_markers(self) -> None:
        text = load_template_text("mistral")
        assert "[INST]" in text
        assert "[/INST]" in text
