"""Repo-level determinism-golden index I/O."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dlm.lock.errors import GoldenIndexSchemaError
from dlm.lock.golden_index import (
    GOLDEN_INDEX_RELATIVE_PATH,
    DeterminismGoldenEntry,
    DeterminismGoldenIndex,
    golden_index_path,
    load_golden_index,
    upsert_golden_index,
    write_golden_index,
)


def _index(*entries: DeterminismGoldenEntry) -> DeterminismGoldenIndex:
    return DeterminismGoldenIndex(
        updated_at=datetime(2026, 4, 22, 4, 25, 32, tzinfo=UTC),
        goldens=entries,
    )


def _entry(
    *,
    golden_relpath: str = "tests/golden/determinism/tuple-0123456789abcdef.json",
    adapter_sha256: str = "a" * 64,
    platform: str = "darwin-arm64",
) -> DeterminismGoldenEntry:
    return DeterminismGoldenEntry(
        golden_relpath=golden_relpath,
        adapter_sha256=adapter_sha256,
        platform=platform,
        pinned_versions={"peft": "0.14.0", "torch": "2.5.1"},
    )


class TestGoldenIndexPath:
    def test_returns_repo_relative_path(self, tmp_path: Path) -> None:
        assert golden_index_path(tmp_path) == tmp_path / GOLDEN_INDEX_RELATIVE_PATH


class TestWriteGoldenIndex:
    def test_writes_readable_json(self, tmp_path: Path) -> None:
        written = write_golden_index(tmp_path, _index(_entry()))
        assert written.is_file()
        text = written.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert text.index('"golden_relpath"') < text.index('"platform"')

    def test_round_trip_equal(self, tmp_path: Path) -> None:
        original = _index(_entry())
        write_golden_index(tmp_path, original)
        loaded = load_golden_index(tmp_path)
        assert loaded == original


class TestLoadGoldenIndex:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_golden_index(tmp_path) is None

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        golden_index_path(tmp_path).parent.mkdir(parents=True)
        golden_index_path(tmp_path).write_text("{not valid", encoding="utf-8")
        with pytest.raises(GoldenIndexSchemaError, match="invalid JSON"):
            load_golden_index(tmp_path)

    def test_non_object_top_level_raises(self, tmp_path: Path) -> None:
        golden_index_path(tmp_path).parent.mkdir(parents=True)
        golden_index_path(tmp_path).write_text("[]", encoding="utf-8")
        with pytest.raises(GoldenIndexSchemaError, match="must be an object"):
            load_golden_index(tmp_path)

    def test_newer_version_is_rejected(self, tmp_path: Path) -> None:
        golden_index_path(tmp_path).parent.mkdir(parents=True)
        golden_index_path(tmp_path).write_text('{"lock_version": 99}', encoding="utf-8")
        with pytest.raises(GoldenIndexSchemaError, match="unsupported lock_version"):
            load_golden_index(tmp_path)


class TestUpsertGoldenIndex:
    def test_creates_index_when_absent(self, tmp_path: Path) -> None:
        upsert_golden_index(
            tmp_path,
            golden_relpath="tests/golden/determinism/tuple-0123456789abcdef.json",
            adapter_sha256="a" * 64,
            platform="darwin-arm64",
            pinned_versions={"torch": "2.5.1", "peft": "0.14.0"},
        )
        loaded = load_golden_index(tmp_path)
        assert loaded is not None
        assert [entry.golden_relpath for entry in loaded.goldens] == [
            "tests/golden/determinism/tuple-0123456789abcdef.json"
        ]

    def test_overwrites_existing_entry_and_sorts(self, tmp_path: Path) -> None:
        write_golden_index(
            tmp_path,
            _index(
                _entry(golden_relpath="tests/golden/determinism/tuple-ffffffffffffffff.json"),
                _entry(
                    golden_relpath="tests/golden/determinism/tuple-aaaaaaaaaaaaaaaa.json",
                    adapter_sha256="b" * 64,
                ),
            ),
        )

        upsert_golden_index(
            tmp_path,
            golden_relpath="tests/golden/determinism/tuple-ffffffffffffffff.json",
            adapter_sha256="c" * 64,
            platform="linux-x86_64",
            pinned_versions={"torch": "2.6.0"},
        )

        loaded = load_golden_index(tmp_path)
        assert loaded is not None
        assert [entry.golden_relpath for entry in loaded.goldens] == [
            "tests/golden/determinism/tuple-aaaaaaaaaaaaaaaa.json",
            "tests/golden/determinism/tuple-ffffffffffffffff.json",
        ]
        assert loaded.goldens[1].adapter_sha256 == "c" * 64
        assert loaded.goldens[1].platform == "linux-x86_64"
