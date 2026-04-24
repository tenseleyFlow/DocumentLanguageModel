"""End-to-end `build_dataset` — parsed sections to split HF Datasets."""

from __future__ import annotations

import pytest

from dlm.data.dataset_builder import build_dataset
from dlm.doc.sections import Section, SectionType


def _s(section_type: SectionType, content: str) -> Section:
    return Section(type=section_type, content=content, start_line=1)


class TestBuildDataset:
    def test_prose_only_document_builds(self) -> None:
        sections = [_s(SectionType.PROSE, f"paragraph {i}") for i in range(20)]
        train, val = build_dataset(sections, seed=42, val_frac=0.1)
        assert len(train) + len(val) == 20
        assert len(val) >= 1

    def test_mixed_document_preserves_shapes(self) -> None:
        sections = [
            _s(SectionType.PROSE, "prose content"),
            _s(SectionType.INSTRUCTION, "### Q\nquestion\n### A\nanswer"),
        ]
        train, val = build_dataset(sections, seed=1, val_frac=0.5)
        all_rows = list(train) + list(val)
        assert len(all_rows) == 2
        # HF Datasets unifies schemas across rows; each row has both keys,
        # but only one will be populated per source row.
        text_rows = [r for r in all_rows if r.get("text")]
        message_rows = [r for r in all_rows if r.get("messages")]
        assert len(text_rows) == 1
        assert len(message_rows) == 1

    def test_reproducible_with_same_seed(self) -> None:
        sections = [_s(SectionType.PROSE, f"row {i}") for i in range(50)]
        t1, v1 = build_dataset(sections, seed=7, val_frac=0.2)
        t2, v2 = build_dataset(sections, seed=7, val_frac=0.2)
        assert list(t1) == list(t2)
        assert list(v1) == list(v2)

    def test_replay_rows_concatenated(self) -> None:
        sections = [_s(SectionType.PROSE, "source doc prose")]
        replay = [
            {"text": "replay-1", "_dlm_section_id": "replay-v1"},
            {"text": "replay-2", "_dlm_section_id": "replay-v2"},
        ]
        train, val = build_dataset(sections, seed=0, val_frac=0.1, replay_rows=replay)
        all_text = {r["text"] for r in list(train) + list(val)}
        assert {"source doc prose", "replay-1", "replay-2"}.issubset(all_text)

    def test_preference_replay_rows_filtered(self) -> None:
        sections = [_s(SectionType.PROSE, "source doc prose")]
        replay = [
            {"text": "sft-replay", "_dlm_section_id": "replay-sft"},
            {
                "prompt": "q",
                "chosen": "good",
                "rejected": "bad",
                "_dlm_section_id": "replay-pref",
            },
        ]
        train, val = build_dataset(sections, seed=0, val_frac=0.1, replay_rows=replay)
        all_rows = list(train) + list(val)
        all_text = {r.get("text") for r in all_rows if r.get("text")}
        assert "sft-replay" in all_text
        assert not any(r.get("prompt") == "q" for r in all_rows)

    def test_empty_rows_raises(self) -> None:
        sections = [_s(SectionType.PROSE, "   ")]
        with pytest.raises(ValueError, match="no trainable rows"):
            build_dataset(sections, seed=0, val_frac=0.1)

    def test_weights_dropping_every_row_raises(self) -> None:
        replay = [
            {
                "text": "replay-only",
                "_dlm_section_id": "replay-v1",
                "_dlm_row_tags": {"lang": "en"},
            },
        ]
        with pytest.raises(ValueError, match="weights dropped every row"):
            build_dataset(
                [],
                seed=0,
                val_frac=0.1,
                replay_rows=replay,
                weights={"lang": {"en": 0.0}},
            )
