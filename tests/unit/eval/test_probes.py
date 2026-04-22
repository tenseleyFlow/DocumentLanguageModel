"""Probe prompt extraction — explicit `!probe` + auto-sample fallback."""

from __future__ import annotations

import dataclasses
import logging

import pytest

from dlm.doc.sections import Section, SectionType
from dlm.eval.probes import Probe, extract_probes


class TestExplicitProbes:
    def test_single_probe(self) -> None:
        body = "### Q !probe\nWhat is Paris?\n### A\nCapital of France."
        s = Section(type=SectionType.INSTRUCTION, content=body)
        probes = extract_probes([s], k=3)
        assert len(probes) == 1
        assert probes[0].prompt == "What is Paris?"
        assert probes[0].reference == "Capital of France."
        assert probes[0].section_id == s.section_id

    def test_multiple_explicit_probes_limited_by_k(self) -> None:
        body = (
            "### Q !probe\nQ1?\n### A\nA1\n\n"
            "### Q !probe\nQ2?\n### A\nA2\n\n"
            "### Q !probe\nQ3?\n### A\nA3"
        )
        s = Section(type=SectionType.INSTRUCTION, content=body)
        probes = extract_probes([s], k=2)
        assert len(probes) == 2
        assert [p.prompt for p in probes] == ["Q1?", "Q2?"]

    def test_non_probe_questions_ignored_when_explicit_present(self) -> None:
        body = "### Q !probe\nexplicit\n### A\nA1\n\n### Q\nnot-probe\n### A\nA2"
        s = Section(type=SectionType.INSTRUCTION, content=body)
        probes = extract_probes([s], k=3)
        assert len(probes) == 2
        # Explicit one comes first.
        assert probes[0].prompt == "explicit"
        # Auto-sampled fills the remainder.
        assert any(p.prompt == "not-probe" for p in probes)


class TestAutoSample:
    def test_auto_sample_when_no_explicit(self) -> None:
        body = "### Q\nQ1?\n### A\nA1\n\n### Q\nQ2?\n### A\nA2\n\n### Q\nQ3?\n### A\nA3"
        s = Section(type=SectionType.INSTRUCTION, content=body)
        probes = extract_probes([s], k=2, seed=42)
        assert len(probes) == 2

    def test_auto_sample_deterministic(self) -> None:
        body = "\n\n".join(f"### Q\nQ{i}?\n### A\nA{i}" for i in range(10))
        s = Section(type=SectionType.INSTRUCTION, content=body)
        a = extract_probes([s], k=3, seed=7)
        b = extract_probes([s], k=3, seed=7)
        assert [p.prompt for p in a] == [p.prompt for p in b]

    def test_different_seeds_yield_different_picks(self) -> None:
        body = "\n\n".join(f"### Q\nQ{i}?\n### A\nA{i}" for i in range(10))
        s = Section(type=SectionType.INSTRUCTION, content=body)
        a = extract_probes([s], k=3, seed=1)
        b = extract_probes([s], k=3, seed=99)
        assert {p.prompt for p in a} != {p.prompt for p in b}

    def test_no_instruction_sections_returns_empty(self) -> None:
        """Prose-only docs have nothing to probe — return [] rather than error."""
        s = Section(type=SectionType.PROSE, content="just prose, no Q/A")
        assert extract_probes([s], k=3) == []

    def test_k_zero_returns_empty(self) -> None:
        body = "### Q !probe\nx\n### A\ny"
        s = Section(type=SectionType.INSTRUCTION, content=body)
        assert extract_probes([s], k=0) == []

    def test_malformed_instruction_logs_warning_once(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        body = "### Q\nunterminated question"
        s = Section(type=SectionType.INSTRUCTION, content=body)
        caplog.set_level(logging.WARNING, logger="dlm.eval.probes")
        assert extract_probes([s], k=3) == []
        assert "probe extraction skipped malformed instruction section" in caplog.text
        assert len(caplog.records) == 1


class TestProbeDataclass:
    def test_probe_is_frozen(self) -> None:
        p = Probe(prompt="hi", reference="hello")
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.prompt = "other"  # type: ignore[misc]
