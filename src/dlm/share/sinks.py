"""Source-string dispatch: map a string to the sink that handles it.

Four schemes recognized today:

- `hf:<org>/<repo>`     → HuggingFace Hub
- `https://...`         → generic HTTPS URL
- `http://...`          → generic HTTPS URL (warns on insecure scheme)
- `peer://host:port/<path>?token=...` → LAN peer (Sprint 28 serve)
- `<local/path>` OR `/abs/path`       → local filesystem

Everything else raises `UnknownSinkError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from dlm.share.errors import UnknownSinkError


class SinkKind(StrEnum):
    HF = "hf"
    URL = "url"
    PEER = "peer"
    LOCAL = "local"


@dataclass(frozen=True)
class SinkSpec:
    """Parsed source string — opaque to callers; dispatched by `kind`."""

    kind: SinkKind
    target: str  # raw suffix after the scheme: "org/repo" / full URL / "host:port/path?..."


def parse_source(source: str) -> SinkSpec:
    """Parse a source string into a `SinkSpec`.

    Examples::

        parse_source("hf:myuser/my-adapter")
        SinkSpec(kind=<SinkKind.HF>, target="myuser/my-adapter")

        parse_source("https://example.com/mydoc.dlm.pack")
        SinkSpec(kind=<SinkKind.URL>, target="https://example.com/mydoc.dlm.pack")

        parse_source("peer://alice-laptop:7337/01HZ...?token=abc")
        SinkSpec(kind=<SinkKind.PEER>, target="alice-laptop:7337/01HZ...?token=abc")

        parse_source("./mydoc.dlm.pack")
        SinkSpec(kind=<SinkKind.LOCAL>, target="./mydoc.dlm.pack")
    """
    if not source:
        raise UnknownSinkError("empty source string")

    if source.startswith("hf:"):
        rest = source[len("hf:") :]
        if not rest or "/" not in rest:
            raise UnknownSinkError(
                f"hf: source must be 'hf:<org>/<repo>', got {source!r}"
            )
        return SinkSpec(kind=SinkKind.HF, target=rest)

    if source.startswith(("http://", "https://")):
        return SinkSpec(kind=SinkKind.URL, target=source)

    if source.startswith("peer://"):
        rest = source[len("peer://") :]
        if not rest:
            raise UnknownSinkError(
                f"peer:// source needs host:port/path, got {source!r}"
            )
        return SinkSpec(kind=SinkKind.PEER, target=rest)

    # Fall through to local path. Accept absolute, relative, and `~`.
    # We don't resolve here — the caller decides when to touch disk.
    path_looking = source.startswith(("/", "./", "../", "~")) or Path(source).suffix
    if path_looking:
        return SinkSpec(kind=SinkKind.LOCAL, target=source)

    raise UnknownSinkError(
        f"unrecognized source {source!r}: expected hf:/https:/peer:/ or a path"
    )
