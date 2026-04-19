"""Typed errors for the replay-corpus subsystem.

The corpus is append-only and content-addressed; corruption is always a
recoverable event (the source `.dlm` is still on disk, and the next
training run can rebuild from scratch) but we want loud, specific
errors rather than raw zstd/cbor/OS exceptions bubbling up into the CLI.
"""

from __future__ import annotations


class ReplayError(Exception):
    """Base for all `dlm.replay` errors."""


class CorpusCorruptError(ReplayError):
    """A zstd frame failed to decompress or decode.

    `byte_offset` is the frame start; `length` is the index entry's
    recorded frame length (the actual bytes we tried to read).
    """

    def __init__(self, message: str, *, byte_offset: int, length: int) -> None:
        super().__init__(message)
        self.byte_offset = byte_offset
        self.length = length


class IndexCorruptError(ReplayError):
    """`index.json` is unreadable, mis-typed, or internally inconsistent."""


class SamplerError(ReplayError):
    """Sampler invariant violated (e.g., `k` exceeds index size, bad scheme)."""
