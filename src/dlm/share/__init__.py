"""Share a `.dlm.pack` across sinks: HF Hub, URL, peer LAN, local file.

Public surface:

- `push(pack_path, destination, **opts)` — upload
- `pull(source, out_dir, **opts)` — download + verify + unpack
- `serve(dlm_path, ...)` — peer LAN endpoint
- `SinkSpec`, `parse_source` — source-string dispatch
- Errors: `ShareError`, `SinkError`, `SignatureError`, `PeerAuthError`,
  `RateLimitError`

Heavy imports (`huggingface_hub`, `requests`, `http.server`) stay
deferred to call sites so `import dlm.share` is cheap.
"""

from __future__ import annotations

from dlm.share.errors import (
    PeerAuthError,
    RateLimitError,
    ShareError,
    SignatureError,
    SinkError,
    UnknownSinkError,
)
from dlm.share.peer import ServeHandle, ServeOptions, serve
from dlm.share.pull import PullResult, pull
from dlm.share.push import PushResult, push
from dlm.share.signing import VerifyResult, VerifyStatus
from dlm.share.sinks import SinkKind, SinkSpec, parse_source

__all__ = [
    "PeerAuthError",
    "PullResult",
    "PushResult",
    "RateLimitError",
    "ServeHandle",
    "ServeOptions",
    "ShareError",
    "SignatureError",
    "SinkError",
    "SinkKind",
    "SinkSpec",
    "UnknownSinkError",
    "VerifyResult",
    "VerifyStatus",
    "parse_source",
    "pull",
    "push",
    "serve",
]
