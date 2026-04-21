"""Typed errors for the share pipeline.

Every CLI-facing failure lands as a `ShareError` subclass so
`dlm.cli.reporter` can render a consistent `share:` prefix.
"""

from __future__ import annotations


class ShareError(Exception):
    """Base for `dlm.share` errors the CLI rewords for users."""


class UnknownSinkError(ShareError):
    """The source string didn't match any registered sink scheme."""


class SinkError(ShareError):
    """A sink rejected the operation (network, auth, protocol).

    Wrap the underlying cause so the CLI can print a short message
    without leaking stack traces from the network library.
    """


class SignatureError(ShareError):
    """A signed pack's signature failed to verify against any trusted key.

    Unsigned packs do NOT raise this — they produce a warning instead.
    Only raised when a signature is present but invalid.
    """


class PeerAuthError(ShareError):
    """An HMAC token failed verification (bad signature, expired, replayed)."""


class RateLimitError(ShareError):
    """Peer-mode rate limit exceeded. Caller returns HTTP 429."""
