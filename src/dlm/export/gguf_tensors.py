"""Minimal GGUF tensor-index reader (Sprint 11.5).

Purpose: locate a named tensor in an emitted GGUF file and read a
single row of its data. We only need two rows (`token_embd.weight`
and `output.weight` at the added-special-token ids) to close the
audit-04 Q2 gap — verifying the adapter's trained embedding rows
match what `convert_hf_to_gguf.py` wrote into the base.

Deliberately narrow scope:

- Metadata walking (plus `general.alignment` lookup) — needed to find
  where the tensor-info block starts and how tensor data is aligned.
- Tensor-info walking — iterates the index and produces
  `TensorEntry(name, shape, dtype, offset)`.
- `read_row(index, name, row_ix)` — reads the raw bytes of one row
  in the tensor's natural dtype. Refuses block-quantized tensor types
  (every k-quant) because a per-row read is not meaningful there.

The outer GGUF file layout we rely on (stable v2+v3):

    b"GGUF" | u32 version | u64 tensor_count | u64 kv_count
    { KV entry } × kv_count
    { tensor_info } × tensor_count
    PAD to `general.alignment`
    { tensor_data_bytes }

`tensor_info`:

    string name
    u32 n_dimensions
    u64[n_dimensions] shape         # row-major, first dim = slowest
    u32 dtype                       # ggml_type enum
    u64 offset                      # relative to tensor_data_start
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from dlm.export._gguf_io import (
    _GGUF_MAGIC,
    _TYPE_UINT32,
    _read_string,
    _read_u32,
    _read_u64,
    _skip_value,
)
from dlm.export.errors import PreflightError

# ggml_type enum slice — we only support the *uncompressed* scalar types
# used for embedding tensors in Sprint 11's emission path. K-quant
# tensors (Q4_K, Q5_K, ...) don't admit a row-sized read without
# dequantizing a block; we refuse them explicitly rather than produce
# meaningless bytes.
GGML_TYPE_F32: Final[int] = 0
GGML_TYPE_F16: Final[int] = 1
GGML_TYPE_BF16: Final[int] = 30
GGML_TYPE_I8: Final[int] = 24
GGML_TYPE_I16: Final[int] = 25
GGML_TYPE_I32: Final[int] = 26
GGML_TYPE_I64: Final[int] = 27
GGML_TYPE_F64: Final[int] = 28

# Bytes-per-element for the scalar types we handle. Any dtype not here
# raises — that covers every block-quantized type plus any future
# additions we haven't vetted.
_SCALAR_BYTES: Final[dict[int, int]] = {
    GGML_TYPE_F32: 4,
    GGML_TYPE_F16: 2,
    GGML_TYPE_BF16: 2,
    GGML_TYPE_I8: 1,
    GGML_TYPE_I16: 2,
    GGML_TYPE_I32: 4,
    GGML_TYPE_I64: 8,
    GGML_TYPE_F64: 8,
}

# Cap on tensor-name length — tensor names are short (`token_embd.weight`
# is 18 chars; the longest real-world names I've seen are under 80).
# 4 KiB is absurd-enough to reject crafted GGUFs that claim a giant
# name string while bounding harmless real-world growth.
_MAX_TENSOR_NAME_BYTES: Final[int] = 4096

# `general.alignment` defaults to 32 in llama.cpp when the metadata
# doesn't specify otherwise. We read the metadata for the actual
# value but fall back to this.
_DEFAULT_ALIGNMENT: Final[int] = 32

# `general.alignment` key is lowercased in recent llama.cpp; earlier
# writers used the same spelling. One string, no variants.
_ALIGNMENT_KEY: Final[str] = "general.alignment"


@dataclass(frozen=True)
class TensorEntry:
    """One row of a GGUF's tensor-info block."""

    name: str
    shape: tuple[int, ...]
    dtype: int  # ggml_type enum
    offset: int  # byte offset from tensor_data_start


@dataclass(frozen=True)
class GgufTensorIndex:
    """Parsed tensor index + enough state to read rows back out."""

    path: Path
    entries: tuple[TensorEntry, ...]
    data_block_start: int  # absolute file offset (post-alignment pad)
    alignment: int

    def find(self, name: str) -> TensorEntry | None:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def row_bytes(self, name: str, row_ix: int) -> bytes:
        """Return the raw bytes of one row of `name`'s tensor.

        Raises `PreflightError` if the tensor doesn't exist, the dtype
        is block-quantized (not row-addressable), or `row_ix` is out
        of range.
        """
        entry = self.find(name)
        if entry is None:
            raise PreflightError(
                probe="gguf_tensor_lookup",
                detail=f"tensor {name!r} not found in {self.path.name}",
            )
        if entry.dtype not in _SCALAR_BYTES:
            raise PreflightError(
                probe="gguf_tensor_dtype",
                detail=(
                    f"tensor {name!r} has ggml dtype {entry.dtype} "
                    "(block-quantized); row-level reads are not supported. "
                    "Embedding tensors are written as F16/F32 pre-quantize; "
                    "this file may have been re-quantized after the fact."
                ),
            )
        if len(entry.shape) < 1:
            raise PreflightError(
                probe="gguf_tensor_shape",
                detail=f"tensor {name!r} has rank 0; nothing to index",
            )
        n_rows = entry.shape[0]
        if row_ix < 0 or row_ix >= n_rows:
            raise PreflightError(
                probe="gguf_tensor_bounds",
                detail=(
                    f"row {row_ix} out of range for {name!r} (shape {entry.shape}, {n_rows} rows)"
                ),
            )
        # GGUF stores tensors row-major with the FIRST dim slowest —
        # shape (vocab_size, hidden) has vocab_size rows of hidden
        # elements each. Row size = prod(shape[1:]) * dtype_bytes.
        row_elems = 1
        for d in entry.shape[1:]:
            row_elems *= d
        row_size = row_elems * _SCALAR_BYTES[entry.dtype]
        row_offset = self.data_block_start + entry.offset + row_ix * row_size
        try:
            with self.path.open("rb") as f:
                f.seek(row_offset)
                raw = f.read(row_size)
        except OSError as exc:
            raise PreflightError(
                probe="gguf_tensor_read",
                detail=f"cannot read row {row_ix} of {name!r}: {exc}",
            ) from exc
        if len(raw) != row_size:
            raise PreflightError(
                probe="gguf_tensor_read",
                detail=(
                    f"short read on row {row_ix} of {name!r}: got "
                    f"{len(raw)} bytes, expected {row_size}"
                ),
            )
        return raw


def load_tensor_index(gguf_path: Path) -> GgufTensorIndex:
    """Parse the header + tensor-info block of `gguf_path`.

    Single pass: walks the KV metadata block (extracting
    `general.alignment`), then iterates the tensor-info block, then
    pads to `alignment` to land on the tensor data block start.

    Raises `PreflightError` on missing file, bad magic, unsupported
    version, or any byte-level inconsistency.
    """
    if not gguf_path.is_file():
        raise PreflightError(
            probe="gguf_tensor_index",
            detail=f"GGUF file {gguf_path} does not exist.",
        )

    try:
        with gguf_path.open("rb") as f:
            magic = f.read(4)
            if magic != _GGUF_MAGIC:
                raise PreflightError(
                    probe="gguf_tensor_index",
                    detail=(
                        f"{gguf_path} does not look like a GGUF file (magic {magic!r} != b'GGUF')."
                    ),
                )
            version = _read_u32(f)
            if version not in (2, 3):
                raise PreflightError(
                    probe="gguf_tensor_index",
                    detail=f"unsupported GGUF version {version}; expected 2 or 3",
                )
            tensor_count = _read_u64(f)
            kv_count = _read_u64(f)

            alignment = _walk_metadata(f, kv_count)
            entries = tuple(_walk_tensor_info(f, tensor_count))

            # After the tensor-info block, pad up to `alignment` so the
            # next byte is the start of the tensor data block.
            pos = f.tell()
            data_block_start = _align_up(pos, alignment)
    except (OSError, struct.error) as exc:
        raise PreflightError(
            probe="gguf_tensor_index",
            detail=f"cannot parse GGUF at {gguf_path}: {exc}",
        ) from exc

    return GgufTensorIndex(
        path=gguf_path,
        entries=entries,
        data_block_start=data_block_start,
        alignment=alignment,
    )


# --- internals ----------------------------------------------------------------


def _walk_metadata(f: Any, kv_count: int) -> int:
    """Skip every KV but capture `general.alignment` if present.

    Returns the effective alignment (default 32 when not specified).
    """
    alignment = _DEFAULT_ALIGNMENT
    for _ in range(kv_count):
        key = _read_string(f)
        value_type = _read_u32(f)
        if key == _ALIGNMENT_KEY and value_type == _TYPE_UINT32:
            alignment = _read_u32(f)
            if alignment <= 0 or alignment & (alignment - 1) != 0:
                raise PreflightError(
                    probe="gguf_tensor_index",
                    detail=(
                        f"invalid general.alignment {alignment}; must be a positive power of two"
                    ),
                )
            continue
        _skip_value(f, value_type)
    return alignment


def _walk_tensor_info(f: Any, tensor_count: int) -> Iterator[TensorEntry]:
    """Iterate the tensor-info block, yielding one `TensorEntry` per row."""
    for _ in range(tensor_count):
        name_len_peek = f.tell()
        # Mirror `_read_string` bounds check but against our tighter
        # tensor-name cap. We can't just call the generic reader
        # because it allows 16 MiB.
        name_len = _read_u64(f)
        if name_len > _MAX_TENSOR_NAME_BYTES:
            raise struct.error(
                f"tensor name length {name_len} at offset {name_len_peek} "
                f"exceeds bound {_MAX_TENSOR_NAME_BYTES}"
            )
        raw = f.read(name_len)
        if len(raw) != name_len:
            raise struct.error("short read in tensor name")
        name = raw.decode("utf-8", errors="replace")

        n_dims = _read_u32(f)
        if n_dims == 0 or n_dims > 8:
            # Real GGUFs are 1D-4D; 8 is already absurd. A nonzero
            # upper bound keeps malformed files from asking for a
            # 2**32-element shape list.
            raise struct.error(f"tensor {name!r} has n_dimensions={n_dims}; refusing")
        shape = tuple(_read_u64(f) for _ in range(n_dims))
        dtype = _read_u32(f)
        offset = _read_u64(f)
        yield TensorEntry(name=name, shape=shape, dtype=dtype, offset=offset)


def _align_up(pos: int, alignment: int) -> int:
    """Round `pos` up to the next multiple of `alignment`."""
    remainder = pos % alignment
    if remainder == 0:
        return pos
    return pos + (alignment - remainder)
