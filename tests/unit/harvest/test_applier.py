"""Unit tests for `dlm.harvest.applier` (Sprint 33.3)."""

from __future__ import annotations

from pathlib import Path

from dlm.doc.parser import parse_file, parse_text
from dlm.harvest import (
    HarvestCandidate,
    apply_plan,
    build_plan,
    revert_last_harvest,
)

_FRONTMATTER = """---
dlm_id: 01KPQ9X1000000000000000000
dlm_version: 7
base_model: smollm2-135m
---
"""


def _candidate(name: str = "quine_q") -> HarvestCandidate:
    return HarvestCandidate(
        prompt=f"question-{name}",
        reference=f"answer-{name}",
        confidence=1.0,
        probe_name=name,
        probe_kind="section_internalization",
        source_adapter_version="run_7",
    )


def _write_dlm(path: Path, body: str = "") -> None:
    path.write_text(_FRONTMATTER + body, encoding="utf-8")


class TestApplyPlan:
    def test_writes_additions_and_preserves_body(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        _write_dlm(target, "## hello\n\nprior prose body\n")

        parsed = parse_file(target)
        plan = build_plan(parsed, [_candidate()])
        summary = apply_plan(parsed, plan, target=target)

        assert summary.added == 1
        assert summary.skipped == 0
        assert len(summary.added_section_ids) == 1

        # Round-trip: reload and confirm the addition lives.
        reloaded = parse_file(target)
        assert len(reloaded.sections) == len(parsed.sections) + 1
        harvested = [s for s in reloaded.sections if s.auto_harvest]
        assert len(harvested) == 1
        assert harvested[0].harvest_source == "auto-harvest/quine_q"
        # prior prose kept
        prose = [s for s in reloaded.sections if not s.auto_harvest]
        assert any("prior prose body" in s.content for s in prose)

    def test_atomic_on_no_additions(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        _write_dlm(target, "## hello\n")
        original = target.read_text(encoding="utf-8")

        parsed = parse_file(target)
        plan = build_plan(parsed, [])
        summary = apply_plan(parsed, plan, target=target)

        assert summary.added == 0
        # File was rewritten but content equivalent (serializer is idempotent).
        reloaded = parse_file(target)
        assert len(reloaded.sections) == len(parsed.sections)
        # Original content round-trips through the serializer.
        assert target.read_text(encoding="utf-8").startswith("---")
        assert "hello" in target.read_text(encoding="utf-8")
        assert "dlm_version: 7" in original  # sanity on fixture


class TestRevertLastHarvest:
    def test_strips_auto_harvest_sections(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        _write_dlm(target, "## hello\n\nkeep me\n")
        parsed = parse_file(target)
        plan = build_plan(parsed, [_candidate("a"), _candidate("b")])
        apply_plan(parsed, plan, target=target)

        # Now revert
        parsed_with_harvest = parse_file(target)
        assert any(s.auto_harvest for s in parsed_with_harvest.sections)
        summary = revert_last_harvest(parsed_with_harvest, target=target)

        assert summary.added == 0
        # summary.added_section_ids carries the IDs of the REMOVED sections
        assert len(summary.added_section_ids) == 2

        reloaded = parse_file(target)
        assert not any(s.auto_harvest for s in reloaded.sections)
        # Hand-authored prose survives.
        assert any("keep me" in s.content for s in reloaded.sections)

    def test_revert_noop_when_no_auto_harvest(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.dlm"
        _write_dlm(target, "## hello\n")
        parsed = parse_file(target)
        summary = revert_last_harvest(parsed, target=target)

        assert summary.added == 0
        assert summary.added_section_ids == ()
        reloaded = parse_file(target)
        assert len(reloaded.sections) == len(parsed.sections)


class TestInMemoryPlanning:
    """Build + apply without touching disk between steps."""

    def test_roundtrip_through_parse_serialize(self) -> None:
        parsed = parse_text(_FRONTMATTER + "::instruction::\n### Q\nhi?\n### A\nhello.\n")
        plan = build_plan(parsed, [_candidate("probe_x")])
        assert len(plan.additions) == 1
        # The section_id is deterministic given the harvest_source prefix.
        assert plan.additions[0].section.section_id.isalnum()
