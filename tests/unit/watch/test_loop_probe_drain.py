"""Watch-loop probe-drain hook — drained probes train alongside file edits."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import DlmFrontmatter
from dlm.doc.sections import Section, SectionType
from dlm.store.manifest import Manifest
from dlm.train.inject import InjectedProbe
from dlm.watch.loop import (
    _probes_to_sections,
    append_injected_probes_audit,
    do_one_cycle,
)


def _section(content: str) -> Section:
    return Section(type=SectionType.PROSE, content=content)


def _parsed(*sections: Section) -> ParsedDlm:
    fm = DlmFrontmatter(
        dlm_id="01KPQTESTZZZZZZZZZZZZZZZZZ",
        dlm_version=7,
        base_model="smollm2-135m",
    )
    return ParsedDlm(frontmatter=fm, sections=tuple(sections), source_path=None)


def _store_stub(manifest: Manifest, tmp_path: Path) -> SimpleNamespace:
    mpath = tmp_path / "manifest.json"
    from dlm.store.manifest import save_manifest

    save_manifest(mpath, manifest)
    return SimpleNamespace(manifest=mpath, logs=tmp_path / "logs")


class TestProbesToSections:
    def test_single_probe_becomes_instruction_section(self) -> None:
        probe = InjectedProbe(prompt="What is X?", reference="Y.", tags=("nightly",))
        sections = _probes_to_sections([probe])
        assert len(sections) == 1
        sec = sections[0]
        assert sec.type is SectionType.INSTRUCTION
        assert sec.auto_harvest is True
        assert sec.harvest_source == "rpc-inject/nightly"
        assert "### Q !probe" in sec.content
        assert "What is X?" in sec.content
        assert "### A\nY." in sec.content

    def test_probe_without_tags_uses_prompt_slug(self) -> None:
        probe = InjectedProbe(prompt="what is DGEMM?", reference="matmul.")
        sections = _probes_to_sections([probe])
        assert sections[0].harvest_source.startswith("rpc-inject/what-is-DGEMM?")

    def test_empty_list_yields_empty_sections(self) -> None:
        assert _probes_to_sections([]) == []


class TestCycleDrainsProbes:
    def test_probes_flow_into_retrain_input(self, tmp_path: Path) -> None:
        """drain_probes' output appends to parsed.sections before diff."""
        existing = _section("alpha")
        parsed = _parsed(existing)
        manifest = Manifest(
            dlm_id="01HZXY",
            base_model="smollm2-135m",
            content_hashes={existing.section_id: existing.section_id},
        )
        store = _store_stub(manifest, tmp_path)

        probe = InjectedProbe(prompt="new-Q", reference="new-A")
        retrain = MagicMock(
            return_value=SimpleNamespace(
                final_train_loss=1.0,
                final_val_loss=0.9,
                steps=10,
                run_id=1,
                adapter_version=1,
            )
        )

        result = do_one_cycle(
            doc_path=tmp_path / "doc.dlm",
            store=store,  # type: ignore[arg-type]
            spec=MagicMock(),
            plan=MagicMock(),
            max_steps=50,
            reload_doc=lambda _p: parsed,  # type: ignore[arg-type,return-value]
            retrain=retrain,
            drain_probes=lambda: [probe],
        )

        assert result.ran is True
        assert result.new_sections == 1  # the probe
        assert result.injected_probes == (probe,)
        # The parsed passed to retrain must include the probe section.
        call_kwargs = retrain.call_args.kwargs if retrain.call_args.kwargs else {}
        call_parsed = retrain.call_args.args[1] if len(retrain.call_args.args) > 1 else None
        assert call_parsed is not None
        # retrain sees both the original prose + the probe-derived instruction.
        types = [s.type for s in call_parsed.sections]
        assert SectionType.PROSE in types
        assert SectionType.INSTRUCTION in types
        _ = call_kwargs  # silence

    def test_skipped_cycle_still_records_injected_probes(self, tmp_path: Path) -> None:
        """When the probe is already in the manifest, the cycle is a skip
        but CycleResult still reports the drained probes — the audit log
        captures them regardless."""
        probe = InjectedProbe(prompt="q", reference="a")
        # Pre-materialize the section so the probe is "unchanged" at diff time.
        probe_section = _probes_to_sections([probe])[0]
        parsed = _parsed()
        manifest = Manifest(
            dlm_id="01HZXY",
            base_model="smollm2-135m",
            content_hashes={probe_section.section_id: probe_section.section_id},
        )
        store = _store_stub(manifest, tmp_path)

        retrain = MagicMock()
        result = do_one_cycle(
            doc_path=tmp_path / "doc.dlm",
            store=store,  # type: ignore[arg-type]
            spec=MagicMock(),
            plan=MagicMock(),
            max_steps=50,
            reload_doc=lambda _p: parsed,  # type: ignore[arg-type,return-value]
            retrain=retrain,
            drain_probes=lambda: [probe],
        )

        assert result.ran is False
        assert result.injected_probes == (probe,)
        retrain.assert_not_called()

    def test_no_drain_hook_yields_empty_injected(self, tmp_path: Path) -> None:
        sec = _section("alpha")
        parsed = _parsed(sec)
        manifest = Manifest(
            dlm_id="01HZXY",
            base_model="smollm2-135m",
            content_hashes={sec.section_id: sec.section_id},
        )
        store = _store_stub(manifest, tmp_path)

        result = do_one_cycle(
            doc_path=tmp_path / "doc.dlm",
            store=store,  # type: ignore[arg-type]
            spec=MagicMock(),
            plan=MagicMock(),
            max_steps=50,
            reload_doc=lambda _p: parsed,  # type: ignore[arg-type,return-value]
            retrain=MagicMock(),
        )
        assert result.injected_probes == ()


class TestAuditLog:
    def test_appends_jsonl_one_line_per_probe(self, tmp_path: Path) -> None:
        store = SimpleNamespace(logs=tmp_path / "logs")
        probes = (
            InjectedProbe(prompt="q1", reference="a1", tags=("ci",)),
            InjectedProbe(prompt="q2", reference="a2"),
        )
        append_injected_probes_audit(store, probes, run_id=7, adapter_version=3)  # type: ignore[arg-type]

        audit_path = tmp_path / "logs" / "rpc-injected.jsonl"
        assert audit_path.exists()
        lines = audit_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        rec1 = json.loads(lines[0])
        assert rec1["prompt"] == "q1"
        assert rec1["tags"] == ["ci"]
        assert rec1["run_id"] == 7
        assert rec1["adapter_version"] == 3
        rec2 = json.loads(lines[1])
        assert rec2["prompt"] == "q2"

    def test_appends_across_calls(self, tmp_path: Path) -> None:
        store = SimpleNamespace(logs=tmp_path / "logs")
        append_injected_probes_audit(
            store,  # type: ignore[arg-type]
            (InjectedProbe(prompt="q1", reference="a1"),),
        )
        append_injected_probes_audit(
            store,  # type: ignore[arg-type]
            (InjectedProbe(prompt="q2", reference="a2"),),
        )
        audit_path = tmp_path / "logs" / "rpc-injected.jsonl"
        assert len(audit_path.read_text(encoding="utf-8").splitlines()) == 2

    def test_empty_probes_no_file_created(self, tmp_path: Path) -> None:
        store = SimpleNamespace(logs=tmp_path / "logs")
        append_injected_probes_audit(store, ())  # type: ignore[arg-type]
        audit_path = tmp_path / "logs" / "rpc-injected.jsonl"
        assert not audit_path.exists()
