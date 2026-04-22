"""Shared GGUF byte-level primitives.

Both `tokenizer_sync` (metadata reader) and `gguf_tensors` (tensor-index
reader) need the same scalar readers + type constants. Kept in a
private module so only one copy exists; the public modules re-export
what their callers need.

Bounds guard: `_read_string` refuses lengths above
`_MAX_STRING_BYTES`. Every caller that reads a length-prefixed array
should also bound the length against a sensible ceiling; we don't
enforce that here because the ceiling depends on caller context
(string vs tensor-name vs token-set) — each caller picks its own.
"""

from __future__ import annotations

import struct
from typing import Any, Final

_GGUF_MAGIC: Final[bytes] = b"GGUF"

# Upper bound on a GGUF metadata string. Tokens / keys / templates all
# fit well under 16 MiB; a crafted file claiming a GB-scale string
# would otherwise drive `f.read(length)` into OOM.
_MAX_STRING_BYTES: Final[int] = 16 * 1024 * 1024

# GGUF metadata value types (stable v2+v3).
_TYPE_UINT8: Final[int] = 0
_TYPE_INT8: Final[int] = 1
_TYPE_UINT16: Final[int] = 2
_TYPE_INT16: Final[int] = 3
_TYPE_UINT32: Final[int] = 4
_TYPE_INT32: Final[int] = 5
_TYPE_FLOAT32: Final[int] = 6
_TYPE_BOOL: Final[int] = 7
_TYPE_STRING: Final[int] = 8
_TYPE_ARRAY: Final[int] = 9
_TYPE_UINT64: Final[int] = 10
_TYPE_INT64: Final[int] = 11
_TYPE_FLOAT64: Final[int] = 12

_FIXED_WIDTH: Final[dict[int, int]] = {
    _TYPE_UINT8: 1,
    _TYPE_INT8: 1,
    _TYPE_UINT16: 2,
    _TYPE_INT16: 2,
    _TYPE_UINT32: 4,
    _TYPE_INT32: 4,
    _TYPE_FLOAT32: 4,
    _TYPE_BOOL: 1,
    _TYPE_UINT64: 8,
    _TYPE_INT64: 8,
    _TYPE_FLOAT64: 8,
}


def _read_u32(f: Any) -> int:
    raw = f.read(4)
    if len(raw) != 4:
        raise struct.error("short read")
    value: int = struct.unpack("<I", raw)[0]
    return value


def _read_u64(f: Any) -> int:
    raw = f.read(8)
    if len(raw) != 8:
        raise struct.error("short read")
    value: int = struct.unpack("<Q", raw)[0]
    return value


def _read_string(f: Any) -> str:
    length = _read_u64(f)
    if length > _MAX_STRING_BYTES:
        raise struct.error(f"GGUF string length {length} exceeds bound {_MAX_STRING_BYTES}")
    raw = f.read(length)
    if len(raw) != length:
        raise struct.error("short read in string")
    decoded: str = raw.decode("utf-8", errors="replace")
    return decoded


def _skip_value(f: Any, value_type: int) -> None:
    """Skip a metadata value of any type without parsing the payload.

    Used by header walkers that locate one key by name and skip
    everything else. Bounds string lengths on the way so a crafted
    file can't drive us into an absurd `seek`.
    """
    if value_type in _FIXED_WIDTH:
        f.seek(_FIXED_WIDTH[value_type], 1)
        return
    if value_type == _TYPE_STRING:
        length = _read_u64(f)
        if length > _MAX_STRING_BYTES:
            raise struct.error(f"GGUF string length {length} exceeds bound {_MAX_STRING_BYTES}")
        f.seek(length, 1)
        return
    if value_type == _TYPE_ARRAY:
        elem_type = _read_u32(f)
        count = _read_u64(f)
        if elem_type in _FIXED_WIDTH:
            f.seek(_FIXED_WIDTH[elem_type] * count, 1)
            return
        if elem_type == _TYPE_STRING:
            for _ in range(count):
                length = _read_u64(f)
                if length > _MAX_STRING_BYTES:
                    raise struct.error(
                        f"GGUF string length {length} exceeds bound {_MAX_STRING_BYTES}"
                    )
                f.seek(length, 1)
            return
        # Nested arrays aren't used by llama.cpp's vocab metadata; treat
        # as unsupported.
        raise struct.error(f"nested/unknown array elem_type {elem_type}")
    raise struct.error(f"unknown GGUF value_type {value_type}")
