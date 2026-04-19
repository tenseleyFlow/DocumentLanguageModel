"""GGUF vocab reader + tokenizer_from_adapter (Sprint 12b)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dlm.export.errors import PreflightError
from dlm.export.tokenizer_sync import (
    assert_gguf_vocab_matches,
    read_gguf_vocab_size,
    tokenizer_from_adapter,
)


# --- GGUF synthesis -------------------------------------------------------

_TYPE_UINT32 = 4
_TYPE_FLOAT32 = 6
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT64 = 10


def _write_string(out: bytearray, s: str) -> None:
    raw = s.encode("utf-8")
    out.extend(struct.pack("<Q", len(raw)))
    out.extend(raw)


def _write_kv_string(out: bytearray, key: str, value: str) -> None:
    _write_string(out, key)
    out.extend(struct.pack("<I", _TYPE_STRING))
    _write_string(out, value)


def _write_kv_u64(out: bytearray, key: str, value: int) -> None:
    _write_string(out, key)
    out.extend(struct.pack("<I", _TYPE_UINT64))
    out.extend(struct.pack("<Q", value))


def _write_kv_string_array(out: bytearray, key: str, values: list[str]) -> None:
    _write_string(out, key)
    out.extend(struct.pack("<I", _TYPE_ARRAY))
    out.extend(struct.pack("<I", _TYPE_STRING))
    out.extend(struct.pack("<Q", len(values)))
    for v in values:
        _write_string(out, v)


def _write_kv_u32_array(out: bytearray, key: str, values: list[int]) -> None:
    _write_string(out, key)
    out.extend(struct.pack("<I", _TYPE_ARRAY))
    out.extend(struct.pack("<I", _TYPE_UINT32))
    out.extend(struct.pack("<Q", len(values)))
    for v in values:
        out.extend(struct.pack("<I", v))


def _make_gguf(tokens: list[str], *, extra_kvs: int = 2) -> bytes:
    """Synthesize a minimal valid GGUF with `tokens` as the tokens array.

    Adds two filler metadata entries (a scalar string + a u32 array) to
    exercise the reader's skip path.
    """
    body = bytearray()
    # Filler string KV ahead of tokens to exercise skip path.
    _write_kv_string(body, "general.name", "test-model")
    # Filler u32 array KV.
    _write_kv_u32_array(body, "foo.bar", [1, 2, 3, 4])
    # The one we want to read.
    _write_kv_string_array(body, "tokenizer.ggml.tokens", tokens)

    header = bytearray()
    header.extend(b"GGUF")
    header.extend(struct.pack("<I", 3))  # version
    header.extend(struct.pack("<Q", 0))  # tensor_count
    header.extend(struct.pack("<Q", 3 if extra_kvs else 1))  # kv_count
    return bytes(header + body)


# --- read_gguf_vocab_size tests ------------------------------------------


class TestReadVocabSize:
    def test_returns_array_length(self, tmp_path: Path) -> None:
        path = tmp_path / "test.gguf"
        path.write_bytes(_make_gguf(["a", "b", "c", "d", "e"]))
        assert read_gguf_vocab_size(path) == 5

    def test_large_vocab(self, tmp_path: Path) -> None:
        vocab = [f"t{i}" for i in range(10_000)]
        path = tmp_path / "big.gguf"
        path.write_bytes(_make_gguf(vocab))
        assert read_gguf_vocab_size(path) == 10_000

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PreflightError, match="does not exist"):
            read_gguf_vocab_size(tmp_path / "nope.gguf")

    def test_bad_magic_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.gguf"
        path.write_bytes(b"NOTAGGUF" + b"\x00" * 20)
        with pytest.raises(PreflightError, match="does not look like a GGUF"):
            read_gguf_vocab_size(path)

    def test_missing_tokens_key_raises(self, tmp_path: Path) -> None:
        """A GGUF with no tokens array at all must be rejected."""
        header = bytearray()
        header.extend(b"GGUF")
        header.extend(struct.pack("<I", 3))
        header.extend(struct.pack("<Q", 0))  # tensor_count
        header.extend(struct.pack("<Q", 1))  # kv_count=1
        _write_kv_string(header, "general.name", "only-name")
        path = tmp_path / "no_tokens.gguf"
        path.write_bytes(bytes(header))
        with pytest.raises(PreflightError, match="tokenizer.ggml.tokens"):
            read_gguf_vocab_size(path)

    def test_truncated_file_raises(self, tmp_path: Path) -> None:
        """A file ending mid-header surfaces as PreflightError."""
        path = tmp_path / "short.gguf"
        path.write_bytes(b"GGUF" + b"\x00\x00")
        with pytest.raises(PreflightError, match="cannot parse GGUF"):
            read_gguf_vocab_size(path)

    def test_tokens_with_non_string_elem_raises(self, tmp_path: Path) -> None:
        """A malformed GGUF where `tokens` is an int array → PreflightError."""
        import struct as _s

        header = bytearray()
        header.extend(b"GGUF")
        header.extend(_s.pack("<I", 3))
        header.extend(_s.pack("<Q", 0))  # tensor_count
        header.extend(_s.pack("<Q", 1))  # kv_count=1
        _write_string(header, "tokenizer.ggml.tokens")
        header.extend(_s.pack("<I", _TYPE_ARRAY))
        header.extend(_s.pack("<I", _TYPE_UINT32))  # wrong elem type
        header.extend(_s.pack("<Q", 3))
        header.extend(_s.pack("<III", 1, 2, 3))

        path = tmp_path / "bad_tokens.gguf"
        path.write_bytes(bytes(header))
        with pytest.raises(PreflightError, match="element type"):
            read_gguf_vocab_size(path)

    def test_unknown_value_type_raises(self, tmp_path: Path) -> None:
        """Value type outside the GGUF spec set bubbles up as a parse error."""
        import struct as _s

        header = bytearray()
        header.extend(b"GGUF")
        header.extend(_s.pack("<I", 3))
        header.extend(_s.pack("<Q", 0))
        header.extend(_s.pack("<Q", 2))  # kv_count=2
        _write_string(header, "bad_kv")
        header.extend(_s.pack("<I", 99))  # unknown type
        header.extend(b"\x00" * 8)  # whatever
        _write_kv_string_array(header, "tokenizer.ggml.tokens", ["a"])
        path = tmp_path / "bad_type.gguf"
        path.write_bytes(bytes(header))
        with pytest.raises(PreflightError, match="cannot parse GGUF"):
            read_gguf_vocab_size(path)

    def test_nested_array_raises(self, tmp_path: Path) -> None:
        """Array-of-array is not supported by llama.cpp's vocab metadata."""
        import struct as _s

        header = bytearray()
        header.extend(b"GGUF")
        header.extend(_s.pack("<I", 3))
        header.extend(_s.pack("<Q", 0))
        header.extend(_s.pack("<Q", 2))
        _write_string(header, "bad_array")
        header.extend(_s.pack("<I", _TYPE_ARRAY))
        header.extend(_s.pack("<I", _TYPE_ARRAY))  # nested array
        header.extend(_s.pack("<Q", 1))
        _write_kv_string_array(header, "tokenizer.ggml.tokens", ["a"])
        path = tmp_path / "nested.gguf"
        path.write_bytes(bytes(header))
        with pytest.raises(PreflightError, match="cannot parse GGUF"):
            read_gguf_vocab_size(path)

    def test_skip_scalar_kv_and_string_array_kv(self, tmp_path: Path) -> None:
        """Exercise `_skip_value` fixed-width scalar + string-array branches."""
        import struct as _s

        body = bytearray()
        # Scalar uint32 KV (exercises the fixed-width skip branch).
        _write_string(body, "general.file_type")
        body.extend(_s.pack("<I", _TYPE_UINT32))
        body.extend(_s.pack("<I", 15))
        # String-array KV (exercises the array-of-strings skip branch).
        _write_kv_string_array(
            body, "tokenizer.ggml.merges", ["a b", "c d", "e f"]
        )
        # Finally the target.
        _write_kv_string_array(body, "tokenizer.ggml.tokens", ["x", "y"])

        header = bytearray()
        header.extend(b"GGUF")
        header.extend(_s.pack("<I", 3))
        header.extend(_s.pack("<Q", 0))
        header.extend(_s.pack("<Q", 3))
        path = tmp_path / "skip.gguf"
        path.write_bytes(bytes(header + body))
        assert read_gguf_vocab_size(path) == 2


# --- assert_gguf_vocab_matches tests -------------------------------------


@dataclass
class _FakeTokenizer:
    vocab: dict[str, int] = field(default_factory=dict)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)


class TestAssertVocabMatches:
    def test_match_passes(self, tmp_path: Path) -> None:
        tokens = ["a", "b", "c"]
        path = tmp_path / "ok.gguf"
        path.write_bytes(_make_gguf(tokens))
        tokenizer = _FakeTokenizer({"a": 0, "b": 1, "c": 2})
        assert_gguf_vocab_matches(path, tokenizer)  # type: ignore[arg-type]

    def test_size_mismatch_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "drift.gguf"
        path.write_bytes(_make_gguf(["a", "b", "c"]))
        tokenizer = _FakeTokenizer({"a": 0, "b": 1})  # only 2 entries
        with pytest.raises(PreflightError, match="does not match GGUF"):
            assert_gguf_vocab_matches(path, tokenizer)  # type: ignore[arg-type]


# --- tokenizer_from_adapter tests ----------------------------------------


class TestTokenizerFromAdapter:
    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PreflightError, match="does not exist"):
            tokenizer_from_adapter(tmp_path / "missing")

    def test_delegates_to_auto_tokenizer_with_local_only(
        self, tmp_path: Path
    ) -> None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        sentinel = object()
        captured: dict[str, Any] = {}

        def fake_from_pretrained(*args: Any, **kwargs: Any) -> Any:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return sentinel

        from transformers import AutoTokenizer

        with patch.object(AutoTokenizer, "from_pretrained", fake_from_pretrained):
            assert tokenizer_from_adapter(adapter) is sentinel
        assert captured["kwargs"]["local_files_only"] is True
        assert captured["kwargs"]["use_fast"] is True

    def test_load_failure_raises_preflight(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()

        def failing_from_pretrained(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("no tokenizer.json here")

        from transformers import AutoTokenizer

        with (
            patch.object(AutoTokenizer, "from_pretrained", failing_from_pretrained),
            pytest.raises(PreflightError, match="cannot load tokenizer"),
        ):
            tokenizer_from_adapter(adapter)
