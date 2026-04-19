"""Tokenizer ↔ GGUF vocab cross-check (Sprint 12b, audit F02/F06).

Sprint 09 persists the *trained* tokenizer (post-bringup, possibly with
an added `<|pad|>`) via `tokenizer.save_pretrained(adapter_dir)` at
training end. That adapter-dir tokenizer is the **source of truth** for
export: Sprint 11's base conversion writes a GGUF whose embedded vocab
MUST match the tokenizer the adapter was trained against, or the
embedding rows for added tokens are either missing or point at the
wrong ids.

Two helpers:

- `tokenizer_from_adapter(adapter_dir)` — loads the HF tokenizer back
  from the directory. Wraps `AutoTokenizer.from_pretrained` with
  `local_files_only=True` so it never touches the network.
- `assert_gguf_vocab_matches(gguf_path, tokenizer)` — parses the
  emitted GGUF's `tokenizer.ggml.tokens` array length and asserts it
  matches `len(tokenizer.get_vocab())`. Raises `PreflightError` with
  the numeric mismatch on drift.

The GGUF parser is inline (~60 lines) rather than taking a dependency
on the vendored `gguf-py` package — that keeps the import path stable
across llama.cpp reorganizations and lets this module be unit-tested
with synthesized tiny GGUF files.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from dlm.export.errors import PreflightError

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


_GGUF_MAGIC: Final[bytes] = b"GGUF"

# Upper bound on a GGUF metadata string — chosen wildly larger than any
# credible real value (tokens are ≤ a few hundred bytes; chat templates
# run tens of KB at most) but small enough to reject a crafted GGUF that
# claims a multi-GB string and drives `f.read(length)` into OOM.
_MAX_STRING_BYTES: Final[int] = 16 * 1024 * 1024

# GGUF value types per llama.cpp's gguf spec (stable v2+v3).
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

# Fixed-size scalar types → byte widths, used to skip arrays of scalars
# without iterating each element.
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

_TOKENS_KEY: Final[str] = "tokenizer.ggml.tokens"


def tokenizer_from_adapter(adapter_dir: Path) -> PreTrainedTokenizerBase:
    """Load the tokenizer saved at training end (Sprint 09).

    `local_files_only=True` forbids network access — the adapter dir is
    the authoritative source. Raises `PreflightError` if the directory
    is missing tokenizer files.
    """
    from transformers import AutoTokenizer

    if not adapter_dir.is_dir():
        raise PreflightError(
            probe="tokenizer_from_adapter",
            detail=f"adapter directory {adapter_dir} does not exist.",
        )
    try:
        tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            str(adapter_dir), local_files_only=True, use_fast=True
        )
    except (OSError, ValueError) as exc:
        raise PreflightError(
            probe="tokenizer_from_adapter",
            detail=f"cannot load tokenizer from {adapter_dir}: {exc}",
        ) from exc
    return tok


def read_gguf_vocab_size(gguf_path: Path) -> int:
    """Return the length of the GGUF's `tokenizer.ggml.tokens` array.

    Parses only enough of the metadata KV block to locate the tokens
    array; all other entries are skipped efficiently. Raises
    `PreflightError` on a missing/unreadable file, an invalid magic,
    or absence of the tokens key.
    """
    if not gguf_path.is_file():
        raise PreflightError(
            probe="gguf_vocab",
            detail=f"GGUF file {gguf_path} does not exist.",
        )

    try:
        with gguf_path.open("rb") as f:
            magic = f.read(4)
            if magic != _GGUF_MAGIC:
                raise PreflightError(
                    probe="gguf_vocab",
                    detail=(
                        f"{gguf_path} does not look like a GGUF file (magic {magic!r} != b'GGUF')."
                    ),
                )
            # version (uint32), tensor_count (uint64), kv_count (uint64)
            _version = _read_u32(f)
            _tensor_count = _read_u64(f)
            kv_count = _read_u64(f)

            for _ in range(kv_count):
                key = _read_string(f)
                value_type = _read_u32(f)
                if key == _TOKENS_KEY and value_type == _TYPE_ARRAY:
                    elem_type = _read_u32(f)
                    count = _read_u64(f)
                    if elem_type != _TYPE_STRING:
                        raise PreflightError(
                            probe="gguf_vocab",
                            detail=(
                                f"{_TOKENS_KEY} has element type {elem_type}, "
                                f"expected string (type {_TYPE_STRING})."
                            ),
                        )
                    return count
                _skip_value(f, value_type)
    except (OSError, struct.error) as exc:
        raise PreflightError(
            probe="gguf_vocab",
            detail=f"cannot parse GGUF at {gguf_path}: {exc}",
        ) from exc

    raise PreflightError(
        probe="gguf_vocab",
        detail=f"{_TOKENS_KEY} key not found in {gguf_path} metadata.",
    )


def assert_gguf_vocab_matches(gguf_path: Path, tokenizer: PreTrainedTokenizerBase) -> None:
    """Raise `PreflightError` if the GGUF vocab size disagrees with the tokenizer.

    Authoritative tokenizer is `len(tokenizer.get_vocab())` — that includes
    base tokens plus any added tokens from the Sprint 07 pad fallback
    path. GGUF vocab comes from the embedded `tokenizer.ggml.tokens` array.
    Equality is the contract; a mismatch means the base converter saw a
    different tokenizer than the one the adapter was trained against.
    """
    tokenizer_vocab = len(tokenizer.get_vocab())
    gguf_vocab = read_gguf_vocab_size(gguf_path)
    if tokenizer_vocab != gguf_vocab:
        raise PreflightError(
            probe="gguf_vocab",
            detail=(
                f"tokenizer vocab ({tokenizer_vocab}) does not match GGUF "
                f"vocab ({gguf_vocab}) for {gguf_path.name}. Re-run base "
                "conversion against the adapter-dir tokenizer."
            ),
        )


# --- internals ------------------------------------------------------------


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
