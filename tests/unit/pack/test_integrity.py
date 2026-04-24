"""`write_checksums` / `verify_checksums` / rollup_sha256 (Sprint 14)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dlm.pack.errors import PackIntegrityError
from dlm.pack.integrity import (
    read_checksums,
    rollup_sha256,
    sha256_of_file,
    verify_checksums,
    write_checksums,
)


def _scratch_tree(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.txt").write_text("alpha\n")
    (root / "b.txt").write_text("beta\n")
    (root / "sub").mkdir()
    (root / "sub" / "c.txt").write_text("gamma\n")
    return root


class TestWriteChecksums:
    def test_covers_all_regular_files(self, tmp_path: Path) -> None:
        root = _scratch_tree(tmp_path)
        checksums = write_checksums(root)
        assert set(checksums) == {"a.txt", "b.txt", "sub/c.txt"}
        assert (root / "CHECKSUMS.sha256").is_file()

    def test_hashes_match_hashlib(self, tmp_path: Path) -> None:
        root = _scratch_tree(tmp_path)
        checksums = write_checksums(root)
        expected = hashlib.sha256((root / "a.txt").read_bytes()).hexdigest()
        assert checksums["a.txt"] == expected

    def test_sorted_lines(self, tmp_path: Path) -> None:
        root = _scratch_tree(tmp_path)
        write_checksums(root)
        lines = (root / "CHECKSUMS.sha256").read_text().splitlines()
        relpaths = [line.split("  ", 1)[1] for line in lines]
        assert relpaths == sorted(relpaths)

    def test_exclude_skips_listed_paths(self, tmp_path: Path) -> None:
        root = _scratch_tree(tmp_path)
        checksums = write_checksums(root, exclude=["a.txt"])
        assert "a.txt" not in checksums

    def test_self_always_excluded(self, tmp_path: Path) -> None:
        """`CHECKSUMS.sha256` itself must never appear in the output."""
        root = _scratch_tree(tmp_path)
        checksums = write_checksums(root)
        assert "CHECKSUMS.sha256" not in checksums


class TestReadChecksums:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PackIntegrityError):
            read_checksums(tmp_path)

    def test_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "CHECKSUMS.sha256").write_text(
            f"\n{'a' * 64}  a.txt\n\n",
            encoding="utf-8",
        )

        assert read_checksums(tmp_path) == {"a.txt": "a" * 64}

    def test_malformed_line_raises(self, tmp_path: Path) -> None:
        (tmp_path / "CHECKSUMS.sha256").write_text("not a valid line\n")
        with pytest.raises(PackIntegrityError):
            read_checksums(tmp_path)

    def test_roundtrip(self, tmp_path: Path) -> None:
        root = _scratch_tree(tmp_path)
        original = write_checksums(root)
        parsed = read_checksums(root)
        assert parsed == original


class TestVerifyChecksums:
    def test_clean_tree_passes(self, tmp_path: Path) -> None:
        root = _scratch_tree(tmp_path)
        write_checksums(root)
        verify_checksums(root)  # no raise

    def test_corrupted_file_raises(self, tmp_path: Path) -> None:
        root = _scratch_tree(tmp_path)
        write_checksums(root)
        (root / "a.txt").write_text("tampered!\n")
        with pytest.raises(PackIntegrityError) as excinfo:
            verify_checksums(root)
        assert excinfo.value.relpath == "a.txt"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        root = _scratch_tree(tmp_path)
        write_checksums(root)
        (root / "b.txt").unlink()
        with pytest.raises(PackIntegrityError) as excinfo:
            verify_checksums(root)
        assert excinfo.value.relpath == "b.txt"


class TestRollup:
    def test_deterministic(self, tmp_path: Path) -> None:
        checksums = {"a": "1" * 64, "b": "2" * 64}
        assert rollup_sha256(checksums) == rollup_sha256(dict(checksums))

    def test_order_independent(self) -> None:
        a = rollup_sha256({"a": "1" * 64, "b": "2" * 64})
        b = rollup_sha256({"b": "2" * 64, "a": "1" * 64})
        assert a == b

    def test_different_inputs_differ(self) -> None:
        a = rollup_sha256({"a": "1" * 64})
        b = rollup_sha256({"a": "2" * 64})
        assert a != b


class TestSha256OfFile:
    def test_chunk_boundaries(self, tmp_path: Path) -> None:
        """File larger than the 1MB chunk size hashes correctly."""
        big = tmp_path / "big.bin"
        big.write_bytes(b"x" * (2 * 1024 * 1024 + 7))
        expected = hashlib.sha256(big.read_bytes()).hexdigest()
        assert sha256_of_file(big) == expected
