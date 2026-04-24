"""GGUF tensor-index reader (Sprint 11.5)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from dlm.export.errors import PreflightError
from dlm.export.gguf_tensors import (
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    TensorEntry,
    _align_up,
    load_tensor_index,
)

# --- GGUF synthesis ---------------------------------------------------------
# Byte-level builder. Kept small + explicit so the tests double as
# documentation of the file layout.

_TYPE_UINT32 = 4
_TYPE_STRING = 8


def _write_string(out: bytearray, s: str) -> None:
    raw = s.encode("utf-8")
    out.extend(struct.pack("<Q", len(raw)))
    out.extend(raw)


def _write_kv_u32(out: bytearray, key: str, value: int) -> None:
    _write_string(out, key)
    out.extend(struct.pack("<I", _TYPE_UINT32))
    out.extend(struct.pack("<I", value))


def _write_kv_string(out: bytearray, key: str, value: str) -> None:
    _write_string(out, key)
    out.extend(struct.pack("<I", _TYPE_STRING))
    _write_string(out, value)


def _write_tensor_info(
    out: bytearray,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: int,
    offset: int,
) -> None:
    _write_string(out, name)
    out.extend(struct.pack("<I", len(shape)))
    for dim in shape:
        out.extend(struct.pack("<Q", dim))
    out.extend(struct.pack("<I", dtype))
    out.extend(struct.pack("<Q", offset))


def _pad_to(out: bytearray, alignment: int) -> int:
    """Zero-pad `out` up to `alignment`; return the post-pad length."""
    while len(out) % alignment:
        out.append(0)
    return len(out)


def _build_gguf(
    *,
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
    alignment: int = 32,
    extra_kvs: int = 1,
) -> bytes:
    """Assemble a valid GGUF with the requested tensors + their data.

    `tensors` is a list of `(name, shape, dtype, data_bytes)`. Offsets
    are computed so each tensor's data packs contiguously starting at
    the data block.
    """
    # Header fields we write: version, tensor_count, kv_count. We'll
    # include one extra KV for general.name + `general.alignment`.
    body = bytearray()
    body.extend(b"GGUF")
    body.extend(struct.pack("<I", 3))  # version
    body.extend(struct.pack("<Q", len(tensors)))  # tensor_count
    kv_count = 1 + extra_kvs  # general.alignment + filler
    body.extend(struct.pack("<Q", kv_count))

    _write_kv_u32(body, "general.alignment", alignment)
    for i in range(extra_kvs):
        _write_kv_string(body, f"filler.{i}", "padding to exercise the skip path")

    # Compute per-tensor offsets (relative to data-block start).
    offsets: list[int] = []
    cursor = 0
    for _name, _shape, _dtype, data in tensors:
        offsets.append(cursor)
        cursor += len(data)

    # Tensor-info block.
    for (name, shape, dtype, _data), offset in zip(tensors, offsets, strict=True):
        _write_tensor_info(body, name=name, shape=shape, dtype=dtype, offset=offset)

    # Pad to alignment.
    _pad_to(body, alignment)

    # Tensor data.
    for _name, _shape, _dtype, data in tensors:
        body.extend(data)

    return bytes(body)


# --- tests ------------------------------------------------------------------


class TestAlignUp:
    def test_at_boundary_returns_pos(self) -> None:
        assert _align_up(64, 32) == 64

    def test_rounds_up(self) -> None:
        assert _align_up(65, 32) == 96

    def test_alignment_1_is_identity(self) -> None:
        assert _align_up(7, 1) == 7


class TestLoadTensorIndex:
    def test_single_tensor_round_trip(self, tmp_path: Path) -> None:
        # 4 rows × 2 cols of F32 = 32 bytes.
        data = b"".join(struct.pack("<f", float(i)) for i in range(8))
        blob = _build_gguf(
            tensors=[("tok_embed.weight", (4, 2), GGML_TYPE_F32, data)],
        )
        path = tmp_path / "one.gguf"
        path.write_bytes(blob)

        index = load_tensor_index(path)
        assert len(index.entries) == 1
        entry = index.entries[0]
        assert entry.name == "tok_embed.weight"
        assert entry.shape == (4, 2)
        assert entry.dtype == GGML_TYPE_F32
        assert entry.offset == 0
        assert index.alignment == 32

    def test_row_bytes_returns_correct_slice(self, tmp_path: Path) -> None:
        # 4 rows × 2 cols of F32, distinctive per-row values.
        rows = [
            struct.pack("<ff", 1.0, 2.0),
            struct.pack("<ff", 3.0, 4.0),
            struct.pack("<ff", 5.0, 6.0),
            struct.pack("<ff", 7.0, 8.0),
        ]
        data = b"".join(rows)
        blob = _build_gguf(
            tensors=[("token_embd.weight", (4, 2), GGML_TYPE_F32, data)],
        )
        path = tmp_path / "rows.gguf"
        path.write_bytes(blob)

        index = load_tensor_index(path)
        for i, expected in enumerate(rows):
            assert index.row_bytes("token_embd.weight", i) == expected

    def test_multi_tensor_offsets(self, tmp_path: Path) -> None:
        # Two F16 tensors; verify the reader honors both offsets.
        a = b"".join(struct.pack("<e", float(i + 1)) for i in range(4))  # 4x1 F16
        b = b"".join(struct.pack("<e", float(i + 10)) for i in range(6))  # 3x2 F16
        blob = _build_gguf(
            tensors=[
                ("token_embd.weight", (4, 1), GGML_TYPE_F16, a),
                ("output.weight", (3, 2), GGML_TYPE_F16, b),
            ],
        )
        path = tmp_path / "two.gguf"
        path.write_bytes(blob)

        index = load_tensor_index(path)
        assert {e.name for e in index.entries} == {
            "token_embd.weight",
            "output.weight",
        }
        # Row 0 of the first tensor is `struct.pack("<e", 1.0)`.
        assert index.row_bytes("token_embd.weight", 0) == struct.pack("<e", 1.0)
        # Row 2 of the second tensor is `struct.pack("<ee", 14.0, 15.0)`.
        assert index.row_bytes("output.weight", 2) == struct.pack("<ee", 14.0, 15.0)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PreflightError, match="does not exist"):
            load_tensor_index(tmp_path / "nope.gguf")

    def test_bad_magic_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.gguf"
        path.write_bytes(b"NOTA" + b"\x00" * 20)
        with pytest.raises(PreflightError, match="does not look like a GGUF"):
            load_tensor_index(path)

    def test_unsupported_version_raises(self, tmp_path: Path) -> None:
        header = bytearray(b"GGUF")
        header.extend(struct.pack("<I", 99))  # version
        header.extend(struct.pack("<Q", 0))
        header.extend(struct.pack("<Q", 0))
        path = tmp_path / "v99.gguf"
        path.write_bytes(bytes(header))
        with pytest.raises(PreflightError, match="unsupported GGUF version 99"):
            load_tensor_index(path)

    def test_bad_alignment_raises(self, tmp_path: Path) -> None:
        # 0 is not a power of two; 3 is not a power of two.
        for bad in (0, 3):
            blob = _build_gguf(
                tensors=[("x", (1,), GGML_TYPE_F32, b"\x00" * 4)],
                alignment=32,  # real build
            )
            # Rewrite the embedded alignment value by hand.
            ba = bytearray(blob)
            # The "general.alignment" key lives at predictable bytes but
            # finding the slot scan-based keeps the test resilient.
            ix = ba.index(b"general.alignment")
            # After the key string: u32 value_type + u32 value
            value_ix = ix + len(b"general.alignment") + 4
            struct.pack_into("<I", ba, value_ix, bad)
            path = tmp_path / f"align{bad}.gguf"
            path.write_bytes(bytes(ba))
            with pytest.raises(PreflightError, match="invalid general.alignment"):
                load_tensor_index(path)

    def test_oversized_tensor_name_refused(self, tmp_path: Path) -> None:
        """A crafted name_len > 4 KiB fails at tensor-info parse time."""
        header = bytearray(b"GGUF")
        header.extend(struct.pack("<I", 3))
        header.extend(struct.pack("<Q", 1))  # tensor_count
        header.extend(struct.pack("<Q", 1))  # kv_count
        _write_kv_u32(header, "general.alignment", 32)
        header.extend(struct.pack("<Q", 1 << 20))  # 1 MiB name length
        path = tmp_path / "oversized.gguf"
        path.write_bytes(bytes(header))
        with pytest.raises(PreflightError, match="cannot parse GGUF"):
            load_tensor_index(path)

    def test_short_tensor_name_read_refused(self, tmp_path: Path) -> None:
        header = bytearray(b"GGUF")
        header.extend(struct.pack("<I", 3))
        header.extend(struct.pack("<Q", 1))  # tensor_count
        header.extend(struct.pack("<Q", 1))  # kv_count
        _write_kv_u32(header, "general.alignment", 32)
        header.extend(struct.pack("<Q", 5))  # claims 5 bytes
        header.extend(b"abc")  # only 3 bytes available
        path = tmp_path / "short-name.gguf"
        path.write_bytes(bytes(header))
        with pytest.raises(PreflightError, match="cannot parse GGUF"):
            load_tensor_index(path)

    @pytest.mark.parametrize("n_dims", [0, 9])
    def test_invalid_tensor_rank_refused(self, tmp_path: Path, n_dims: int) -> None:
        header = bytearray(b"GGUF")
        header.extend(struct.pack("<I", 3))
        header.extend(struct.pack("<Q", 1))  # tensor_count
        header.extend(struct.pack("<Q", 1))  # kv_count
        _write_kv_u32(header, "general.alignment", 32)
        _write_string(header, "token_embd.weight")
        header.extend(struct.pack("<I", n_dims))
        path = tmp_path / f"ndims-{n_dims}.gguf"
        path.write_bytes(bytes(header))
        with pytest.raises(PreflightError, match="cannot parse GGUF"):
            load_tensor_index(path)


class TestRowBytesErrors:
    def _build_basic(self, tmp_path: Path) -> Path:
        # Vocab=2, hidden=3 F16 tensor.
        data = struct.pack("<6e", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        blob = _build_gguf(tensors=[("token_embd.weight", (2, 3), GGML_TYPE_F16, data)])
        path = tmp_path / "basic.gguf"
        path.write_bytes(blob)
        return path

    def test_unknown_tensor_raises(self, tmp_path: Path) -> None:
        index = load_tensor_index(self._build_basic(tmp_path))
        with pytest.raises(PreflightError, match="not found"):
            index.row_bytes("absent.weight", 0)

    def test_out_of_range_row_raises(self, tmp_path: Path) -> None:
        index = load_tensor_index(self._build_basic(tmp_path))
        with pytest.raises(PreflightError, match="out of range"):
            index.row_bytes("token_embd.weight", 5)

    def test_negative_row_raises(self, tmp_path: Path) -> None:
        index = load_tensor_index(self._build_basic(tmp_path))
        with pytest.raises(PreflightError, match="out of range"):
            index.row_bytes("token_embd.weight", -1)

    def test_block_quantized_dtype_refused(self, tmp_path: Path) -> None:
        """A k-quant embedding (dtype 14 = Q4_K) must refuse row reads."""
        # Build using dtype=14 (Q4_K) with arbitrary data; load succeeds
        # because the index walk doesn't care about dtype, but row_bytes
        # refuses.
        data = b"\x00" * 16  # ignored — row_bytes never reads it
        blob = _build_gguf(tensors=[("token_embd.weight", (2, 4), 14, data)])
        path = tmp_path / "kq.gguf"
        path.write_bytes(blob)
        index = load_tensor_index(path)
        with pytest.raises(PreflightError, match="block-quantized"):
            index.row_bytes("token_embd.weight", 0)

    def test_rank_zero_tensor_refused(self, tmp_path: Path) -> None:
        index = load_tensor_index(self._build_basic(tmp_path))
        index = index.__class__(
            path=index.path,
            entries=(
                TensorEntry(name="token_embd.weight", shape=(), dtype=GGML_TYPE_F16, offset=0),
            ),
            data_block_start=index.data_block_start,
            alignment=index.alignment,
        )
        with pytest.raises(PreflightError, match="rank 0"):
            index.row_bytes("token_embd.weight", 0)

    def test_short_row_read_raises(self, tmp_path: Path) -> None:
        path = self._build_basic(tmp_path)
        index = load_tensor_index(path)
        path.write_bytes(path.read_bytes()[:-1])
        with pytest.raises(PreflightError, match="short read on row 1"):
            index.row_bytes("token_embd.weight", 1)

    def test_oserror_while_opening_tensor_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = self._build_basic(tmp_path)
        index = load_tensor_index(path)
        original_open = Path.open

        def _boom(self: Path, *args: object, **kwargs: object) -> object:
            if self == index.path:
                raise OSError("nope")
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", _boom)
        with pytest.raises(PreflightError, match="cannot read row 0"):
            index.row_bytes("token_embd.weight", 0)


class TestFindApi:
    def test_find_returns_entry_or_none(self, tmp_path: Path) -> None:
        data = struct.pack("<4e", 1.0, 2.0, 3.0, 4.0)
        blob = _build_gguf(tensors=[("a.weight", (2, 2), GGML_TYPE_F16, data)])
        path = tmp_path / "find.gguf"
        path.write_bytes(blob)
        index = load_tensor_index(path)
        found = index.find("a.weight")
        assert isinstance(found, TensorEntry)
        assert found.name == "a.weight"
        assert index.find("missing") is None
