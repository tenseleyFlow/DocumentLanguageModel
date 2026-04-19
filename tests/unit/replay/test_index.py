"""JSON index load/save — atomic, sorted, typed."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.replay.errors import IndexCorruptError
from dlm.replay.index import load_index, save_index
from dlm.replay.models import IndexEntry


def _entry(sid: str, offset: int = 0, length: int = 100) -> IndexEntry:
    return IndexEntry(section_id=sid, byte_offset=offset, length=length)


class TestLoadIndex:
    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        assert load_index(tmp_path / "absent.json") == []

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "index.json"
        p.write_bytes(b"not valid json {{{")
        with pytest.raises(IndexCorruptError, match="JSON"):
            load_index(p)

    def test_wrong_shape_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "index.json"
        p.write_bytes(b'[{"section_id": "short", "byte_offset": 0, "length": 10}]')
        with pytest.raises(IndexCorruptError, match="invalid entries"):
            load_index(p)

    def test_read_error_surfaces_as_corrupt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OSError during read_bytes is wrapped into IndexCorruptError."""
        p = tmp_path / "index.json"
        p.write_bytes(b"[]")

        def _boom(self: object) -> bytes:
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "read_bytes", _boom)
        with pytest.raises(IndexCorruptError, match="cannot read"):
            load_index(p)


class TestSaveIndex:
    def test_round_trip(self, tmp_path: Path) -> None:
        entries = [_entry("a" * 16), _entry("b" * 16)]
        save_index(tmp_path / "index.json", entries)
        back = load_index(tmp_path / "index.json")
        assert len(back) == 2

    def test_sorted_by_section_id(self, tmp_path: Path) -> None:
        entries = [_entry("c" * 16), _entry("a" * 16), _entry("b" * 16)]
        p = tmp_path / "index.json"
        save_index(p, entries)
        loaded = load_index(p)
        assert [e.section_id for e in loaded] == ["a" * 16, "b" * 16, "c" * 16]

    def test_byte_identical_across_saves(self, tmp_path: Path) -> None:
        """Same entries in different order produce identical file bytes."""
        entries_a = [_entry("a" * 16, 0, 50), _entry("b" * 16, 100, 50)]
        entries_b = [_entry("b" * 16, 100, 50), _entry("a" * 16, 0, 50)]
        pa = tmp_path / "a.json"
        pb = tmp_path / "b.json"
        save_index(pa, entries_a)
        save_index(pb, entries_b)
        assert pa.read_bytes() == pb.read_bytes()
