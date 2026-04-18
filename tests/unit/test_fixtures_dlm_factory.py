"""Verify dlm_factory produces structurally-valid .dlm text.

Until Sprint 03's parser lands, we assert shape (frontmatter delimiters,
section fences, Q/A markers). The parser tests in Sprint 03 will
round-trip these blobs as the real acceptance check.
"""

from __future__ import annotations

import re

from tests.fixtures.dlm_factory import instruction, make_dlm, preference, prose


class TestMakeDlm:
    def test_default_has_frontmatter_and_body(self) -> None:
        text = make_dlm()
        assert text.startswith("---\n")
        fm_end = text.index("\n---\n", 4)
        body = text[fm_end + len("\n---\n") :]
        assert body.strip(), "body must not be empty"

    def test_explicit_dlm_id_is_preserved(self) -> None:
        text = make_dlm(dlm_id="01HZ000000000000000000000X")
        assert "dlm_id: 01HZ000000000000000000000X" in text

    def test_base_model_reflects_arg(self) -> None:
        text = make_dlm(base_model="hf:org/custom-model")
        assert (
            "base_model: hf:org/custom-model" in text or 'base_model: "hf:org/custom-model"' in text
        )

    def test_instruction_section_emits_q_a_pairs(self) -> None:
        text = make_dlm(
            sections=[instruction(("Q1?", "A1."), ("Q2?", "A2."))],
        )
        assert "::instruction::" in text
        # Count Q/A headers — must match pair count.
        assert text.count("### Q") == 2
        assert text.count("### A") == 2
        assert "Q1?" in text
        assert "A1." in text
        assert "Q2?" in text
        assert "A2." in text

    def test_preference_section_emits_triples(self) -> None:
        text = make_dlm(
            sections=[preference(("prompt", "good", "bad"))],
        )
        assert "::preference::" in text
        assert "### Prompt" in text
        assert "### Chosen" in text
        assert "### Rejected" in text
        for piece in ("prompt", "good", "bad"):
            assert piece in text

    def test_prose_section_is_verbatim(self) -> None:
        body = "# heading\n\nparagraph with **markdown**.\n"
        text = make_dlm(sections=[prose(body)])
        assert body in text

    def test_mixed_sections_in_order(self) -> None:
        text = make_dlm(
            sections=[
                prose("intro.\n"),
                instruction(("q", "a")),
                preference(("p", "c", "r")),
            ],
        )
        # Assert positional order via index.
        i_intro = text.index("intro.")
        i_instr = text.index("::instruction::")
        i_pref = text.index("::preference::")
        assert i_intro < i_instr < i_pref

    def test_training_override_appears_in_frontmatter(self) -> None:
        text = make_dlm(training_overrides={"lora_r": 16, "learning_rate": 1e-3})
        assert re.search(r"^\s*lora_r:\s*16$", text, re.MULTILINE)
        assert re.search(r"^\s*learning_rate:\s*0\.001$", text, re.MULTILINE)

    def test_system_prompt_emitted_as_block_scalar(self) -> None:
        text = make_dlm(system_prompt="line one\nline two")
        assert "system_prompt: |\n  line one\n  line two" in text

    def test_generated_output_ends_with_newline(self) -> None:
        text = make_dlm()
        assert text.endswith("\n")

    def test_no_explicit_id_generates_ulid(self) -> None:
        text_a = make_dlm()
        text_b = make_dlm()
        # Two calls should produce different ULIDs.
        id_a = re.search(r"^dlm_id:\s*(\S+)$", text_a, re.MULTILINE)
        id_b = re.search(r"^dlm_id:\s*(\S+)$", text_b, re.MULTILINE)
        assert id_a is not None
        assert id_b is not None
        assert id_a.group(1) != id_b.group(1)
        assert len(id_a.group(1)) == 26  # ULID canonical length
