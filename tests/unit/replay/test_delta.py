"""Content-delta classification against a previous manifest."""

from __future__ import annotations

from dlm.doc.sections import Section, SectionType
from dlm.replay.delta import diff_against_manifest
from dlm.store.manifest import Manifest


def _manifest(content_hashes: dict[str, str]) -> Manifest:
    return Manifest(
        dlm_id="01KTEST",
        base_model="qwen2.5-1.5b",
        content_hashes=content_hashes,
    )


class TestDiffAgainstManifest:
    def test_all_new_on_empty_manifest(self) -> None:
        s1 = Section(type=SectionType.PROSE, content="hello")
        s2 = Section(type=SectionType.PROSE, content="world")
        cs = diff_against_manifest([s1, s2], _manifest({}))
        assert {s.section_id for s in cs.new} == {s1.section_id, s2.section_id}
        assert cs.unchanged == []
        assert cs.removed == []
        assert cs.changed == []

    def test_unchanged_classified_correctly(self) -> None:
        s1 = Section(type=SectionType.PROSE, content="hello")
        m = _manifest({s1.section_id: s1.section_id})
        cs = diff_against_manifest([s1], m)
        assert cs.new == []
        assert [s.section_id for s in cs.unchanged] == [s1.section_id]

    def test_removed_populated_from_manifest(self) -> None:
        s1 = Section(type=SectionType.PROSE, content="kept")
        ghost_sid = "deadbeefdeadbeef"
        m = _manifest({s1.section_id: s1.section_id, ghost_sid: ghost_sid})
        cs = diff_against_manifest([s1], m)
        assert cs.removed == [ghost_sid]

    def test_changed_always_empty_under_current_design(self) -> None:
        """Content-addressed sections: a content edit is new+removed, not changed."""
        old = Section(type=SectionType.PROSE, content="v1")
        new = Section(type=SectionType.PROSE, content="v2")
        m = _manifest({old.section_id: old.section_id})
        cs = diff_against_manifest([new], m)
        assert cs.changed == []
        assert [s.section_id for s in cs.new] == [new.section_id]
        assert cs.removed == [old.section_id]

    def test_duplicate_content_in_current_doc_dedupes(self) -> None:
        s1 = Section(type=SectionType.PROSE, content="repeat")
        s2 = Section(type=SectionType.PROSE, content="repeat")
        cs = diff_against_manifest([s1, s2], _manifest({}))
        # Both have the same section_id — only classified once as "new".
        assert len(cs.new) == 1

    def test_section_type_change_is_new_plus_removed(self) -> None:
        """Type is part of the hash; a type change produces different ids."""
        prose = Section(type=SectionType.PROSE, content="same text")
        instr = Section(type=SectionType.INSTRUCTION, content="same text")
        assert prose.section_id != instr.section_id
        m = _manifest({prose.section_id: prose.section_id})
        cs = diff_against_manifest([instr], m)
        assert [s.section_id for s in cs.new] == [instr.section_id]
        assert cs.removed == [prose.section_id]
