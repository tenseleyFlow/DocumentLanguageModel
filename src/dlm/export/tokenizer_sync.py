"""Tokenizer ↔ GGUF vocab cross-check.

Training persists the *trained* tokenizer (post-bringup, possibly with
an added `<|pad|>`) via `tokenizer.save_pretrained(adapter_dir)` at
training end. That adapter-dir tokenizer is the **source of truth** for
export: base conversion writes a GGUF whose embedded vocab
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
from typing import TYPE_CHECKING, Final

from dlm.export._gguf_io import (
    _GGUF_MAGIC,
    _TYPE_ARRAY,
    _TYPE_STRING,
    _read_string,
    _read_u32,
    _read_u64,
    _skip_value,
)
from dlm.export.errors import PreflightError

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_TOKENS_KEY: Final[str] = "tokenizer.ggml.tokens"


def tokenizer_from_adapter(adapter_dir: Path) -> PreTrainedTokenizerBase:
    """Load the tokenizer saved at training end.

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
    base tokens plus any added tokens from the pad fallback
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
# Byte-level primitives (`_read_u32`, `_read_u64`, `_read_string`,
# `_skip_value`) live in `dlm.export._gguf_io` and are imported at the
# top of this module. `gguf_tensors` uses the same set.
