"""Audit F15: UTF-8 strict, BOM strip, CRLF normalization."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from dlm.io.text import (
    DlmEncodingError,
    normalize_for_hashing,
    read_text,
    read_text_str,
    write_text,
)


class TestReadText:
    def test_utf8_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "sample.txt"
        p.write_bytes(b"hello, world\n")
        assert read_text(p) == "hello, world\n"

    def test_bom_is_stripped_and_warned(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        p = tmp_path / "bom.txt"
        p.write_bytes(b"\xef\xbb\xbfhello\n")
        with caplog.at_level(logging.WARNING, logger="dlm.io.text"):
            text = read_text(p)
        assert text == "hello\n"
        assert any("BOM" in rec.message for rec in caplog.records)

    def test_crlf_is_normalized(self, tmp_path: Path) -> None:
        p = tmp_path / "windows.txt"
        p.write_bytes(b"line1\r\nline2\r\n")
        assert read_text(p) == "line1\nline2\n"

    def test_lone_cr_is_normalized(self, tmp_path: Path) -> None:
        p = tmp_path / "classic_mac.txt"
        p.write_bytes(b"line1\rline2\r")
        assert read_text(p) == "line1\nline2\n"

    def test_mixed_endings_normalized(self, tmp_path: Path) -> None:
        p = tmp_path / "mixed.txt"
        p.write_bytes(b"a\r\nb\nc\r")
        assert read_text(p) == "a\nb\nc\n"

    def test_invalid_utf8_raises_with_offset(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.txt"
        # \xff is never valid as a start byte in UTF-8
        p.write_bytes(b"ok\xffbad")
        with pytest.raises(DlmEncodingError) as exc_info:
            read_text(p)
        assert exc_info.value.byte_offset == 2
        assert exc_info.value.path == p


class TestReadTextStr:
    def test_bytes_roundtrip(self) -> None:
        assert read_text_str(b"hello\n") == "hello\n"

    def test_invalid_raises_with_source(self) -> None:
        with pytest.raises(DlmEncodingError) as exc_info:
            read_text_str(b"\xff", source="fixture:broken")
        assert "fixture:broken" in str(exc_info.value)


class TestWriteText:
    def test_writes_utf8_lf_no_bom(self, tmp_path: Path) -> None:
        p = tmp_path / "out.txt"
        write_text(p, "line1\r\nline2\r\n")
        raw = p.read_bytes()
        assert raw == b"line1\nline2\n"

    def test_write_is_atomic_leaves_no_tmp(self, tmp_path: Path) -> None:
        p = tmp_path / "out.txt"
        write_text(p, "content\n")
        siblings = list(tmp_path.iterdir())
        assert siblings == [p], "write_text must clean up temp files"


class TestNormalizeForHashing:
    def test_bom_stripped(self) -> None:
        assert normalize_for_hashing("\ufeffhello") == "hello"

    def test_crlf_normalized(self) -> None:
        assert normalize_for_hashing("a\r\nb\rc") == "a\nb\nc"

    def test_hash_identity_across_platforms(self) -> None:
        """Windows-style CRLF and Unix-style LF with same content must hash-identically.

        This is the core Sprint 03 / audit F15 contract: section IDs are stable
        under line-ending edits.
        """
        unix = "header\n\nbody line 1\nbody line 2\n"
        windows = "header\r\n\r\nbody line 1\r\nbody line 2\r\n"
        assert normalize_for_hashing(unix) == normalize_for_hashing(windows)
