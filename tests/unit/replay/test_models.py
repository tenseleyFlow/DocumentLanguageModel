"""Pydantic model validation for SectionSnapshot + IndexEntry."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlm.replay.models import IndexEntry, SectionSnapshot


class TestSectionSnapshot:
    def test_valid_snapshot_accepted(self) -> None:
        s = SectionSnapshot(section_id="a" * 16, section_type="prose", content="hello")
        assert s.section_id == "a" * 16
        assert s.section_type == "prose"

    def test_bad_section_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="16-char"):
            SectionSnapshot(section_id="abc", section_type="prose", content="x")
        with pytest.raises(ValidationError, match="16-char"):
            SectionSnapshot(section_id="A" * 16, section_type="prose", content="x")  # uppercase
        with pytest.raises(ValidationError, match="16-char"):
            SectionSnapshot(section_id="g" * 16, section_type="prose", content="x")  # non-hex

    def test_unknown_section_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SectionSnapshot(section_id="a" * 16, section_type="unknown", content="x")  # type: ignore[arg-type]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SectionSnapshot.model_validate(
                {
                    "section_id": "a" * 16,
                    "section_type": "prose",
                    "content": "x",
                    "surprise": "field",
                }
            )

    def test_frozen(self) -> None:
        s = SectionSnapshot(section_id="a" * 16, section_type="prose", content="x")
        with pytest.raises(ValidationError):
            s.content = "y"  # type: ignore[misc]


class TestIndexEntry:
    def test_valid_entry(self) -> None:
        e = IndexEntry(section_id="a" * 16, byte_offset=0, length=100)
        assert e.weight == 1.0

    def test_negative_offset_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IndexEntry(section_id="a" * 16, byte_offset=-1, length=100)

    def test_zero_length_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IndexEntry(section_id="a" * 16, byte_offset=0, length=0)

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IndexEntry(section_id="a" * 16, byte_offset=0, length=1, weight=-0.5)
