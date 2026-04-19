"""ULID minter — 26-char Crockford base32 sortable identifier.

Implements ULID (Universally Unique Lexicographically Sortable
Identifier) per the spec at https://github.com/ulid/spec: 48 bits of
millisecond timestamp + 80 bits of entropy, encoded in Crockford
base32 (excluding `I`, `L`, `O`, `U` — ambiguous glyphs).

We roll this in ~40 lines rather than adding a `ulid-py` dependency
— the format is small and stable, and ULIDs are generated only at
`dlm init` time (single call per document, not in a hot path).

The encoding matches `DlmFrontmatter._validate_ulid`: upper-case,
exactly 26 characters, alphabet `[0-9A-HJKMNP-TV-Z]`.
"""

from __future__ import annotations

import os
import time
from typing import Final

# Crockford base32 alphabet: 0-9, A-Z excluding I, L, O, U.
_ALPHABET: Final[str] = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def mint_ulid() -> str:
    """Return a fresh 26-char Crockford base32 ULID.

    Time prefix is the current UTC millisecond count (48 bits); the
    remaining 80 bits come from `os.urandom`. Two calls in the same
    millisecond produce distinct ULIDs with overwhelming probability
    (2^80 entropy slots).
    """
    timestamp_ms = int(time.time() * 1000)
    random_bytes = os.urandom(10)
    # Concatenate 6 bytes of timestamp (big-endian) + 10 bytes of random.
    payload = timestamp_ms.to_bytes(6, "big") + random_bytes
    return _encode_crockford(payload)


def _encode_crockford(data: bytes) -> str:
    """Encode 16 bytes (128 bits) as 26 Crockford base32 chars.

    ULID uses a non-padded encoding: 128 bits → 26 characters because
    `ceil(128/5) == 26`. The top bits of the first character carry
    the two-bit remainder; the ULID spec constrains them to 0-7
    (first char is `0..7` Crockford → digits only).
    """
    if len(data) != 16:
        raise ValueError(f"ULID payload must be 16 bytes, got {len(data)}")

    # Convert to a single 128-bit int.
    value = int.from_bytes(data, "big")

    # Encode from the least-significant 5-bit group upward.
    chars: list[str] = []
    for _ in range(26):
        chars.append(_ALPHABET[value & 0x1F])
        value >>= 5

    return "".join(reversed(chars))
