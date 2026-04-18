"""UTF-8 text I/O with BOM and CRLF hygiene (audit F15).

Every sprint that reads or writes a `.dlm` file or any plain-text store
artifact must route through these helpers. The contract:

- UTF-8 strict: invalid bytes raise `DlmEncodingError` with the offending
  byte offset.
- A leading UTF-8 BOM is stripped and a warning is emitted (some Windows
  editors add it; YAML parsers choke on it).
- CRLF is normalized to LF before hashing or parsing, so section IDs are
  stable across Windows and Unix edits of the same content.

Binary I/O uses the normal `open(..., "rb")` path; this module is only for
text.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

_LOG: Final = logging.getLogger(__name__)

_UTF8_BOM: Final = "\ufeff"
_UTF8_BOM_BYTES: Final = b"\xef\xbb\xbf"


class DlmEncodingError(ValueError):
    """Raised when a text file is not valid UTF-8.

    `byte_offset` is the zero-based position of the first offending byte,
    as reported by the codec.
    """

    def __init__(self, path: Path | None, byte_offset: int, reason: str) -> None:
        self.path = path
        self.byte_offset = byte_offset
        self.reason = reason
        where = str(path) if path is not None else "<text>"
        super().__init__(f"{where}: invalid UTF-8 at byte {byte_offset}: {reason}")


def read_text(path: Path) -> str:
    """Read `path` as UTF-8; strip BOM; normalize CRLF → LF.

    Raises `DlmEncodingError` on invalid UTF-8. Emits a warning via the
    `dlm.io.text` logger when a BOM is present so pipelines surface the
    (usually Windows-editor) provenance.
    """
    raw = path.read_bytes()
    return _decode(raw, path=path)


def read_text_str(raw: bytes, *, source: str = "<bytes>") -> str:
    """In-memory variant of `read_text` for tests and streamed input."""
    # Decode with path=None; `source` is only for the error message.
    try:
        return _decode(raw, path=None)
    except DlmEncodingError as exc:
        # Rewrite to include source in the message.
        raise DlmEncodingError(None, exc.byte_offset, f"{source}: {exc.reason}") from exc


def write_text(path: Path, content: str) -> None:
    """Write `content` as UTF-8 with LF line endings, no BOM.

    Writes atomically: writes to a temp sibling file then `os.replace`s.
    """
    # Normalize line endings on the way out too, belt-and-braces.
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    tmp = path.with_suffix(path.suffix + f".tmp.{_pid()}")
    tmp.write_bytes(normalized.encode("utf-8"))
    tmp.replace(path)


def normalize_for_hashing(content: str) -> str:
    """Produce the canonical form used when hashing text content.

    - BOM removed (if any)
    - CRLF → LF
    - CR alone → LF

    This must exactly mirror what `_decode` produces so that
    `hash(read_text(path)) == hash(normalize_for_hashing(serialize(...)))`.
    """
    if content.startswith(_UTF8_BOM):
        content = content[1:]
    return content.replace("\r\n", "\n").replace("\r", "\n")


# --- internals ---


def _decode(raw: bytes, *, path: Path | None) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DlmEncodingError(path, exc.start, exc.reason) from exc

    if text.startswith(_UTF8_BOM):
        where = str(path) if path is not None else "<text>"
        _LOG.warning("%s: UTF-8 BOM present; stripped", where)
        text = text[1:]

    # CRLF first, then stray CR, to avoid double-replacing \r\n as \r then \n.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _pid() -> int:
    import os as _os

    return _os.getpid()
