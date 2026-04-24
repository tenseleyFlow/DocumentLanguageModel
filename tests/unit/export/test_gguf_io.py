"""Private GGUF IO helper coverage."""

from __future__ import annotations

import io
import struct

import pytest

from dlm.export._gguf_io import _TYPE_ARRAY, _TYPE_STRING, _read_string, _read_u64, _skip_value


def test_read_u64_short_read_raises() -> None:
    with pytest.raises(struct.error, match="short read"):
        _read_u64(io.BytesIO(b"\x01\x02"))


def test_read_string_short_read_raises() -> None:
    data = io.BytesIO(struct.pack("<Q", 4) + b"ab")

    with pytest.raises(struct.error, match="short read in string"):
        _read_string(data)


def test_skip_value_string_array_huge_length_raises() -> None:
    data = io.BytesIO(
        struct.pack("<I", _TYPE_STRING)
        + struct.pack("<Q", 1)
        + struct.pack("<Q", (16 * 1024 * 1024) + 1)
    )

    with pytest.raises(struct.error, match="exceeds bound"):
        _skip_value(data, _TYPE_ARRAY)
