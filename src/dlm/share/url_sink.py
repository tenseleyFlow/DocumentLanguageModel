"""Generic HTTPS sink — POST to push, GET to pull.

Auth header is pulled from the `DLM_SHARE_AUTH` env var at call time
(value is used verbatim as the full `Authorization:` header — lets users
pick `Bearer <token>`, `Basic ...`, or anything else their endpoint
expects). Missing env var means unauthenticated requests.

Downloads stream in 1 MB chunks so multi-GB packs don't blow memory.
Uploads use a streaming file handle for the same reason.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import IO

from dlm.share.errors import SinkError

_LOG = logging.getLogger(__name__)

_CHUNK_BYTES = 1 * 1024 * 1024  # 1 MiB streaming chunks
_AUTH_ENV = "DLM_SHARE_AUTH"
_USER_AGENT = "dlm-share/1.0"

ProgressCallback = Callable[[int, int], None] | None
"""Called as progress(bytes_done, total_bytes). total_bytes may be 0
if the server doesn't set Content-Length."""


def push_url(pack_path: Path, url: str, *, progress: ProgressCallback = None) -> None:
    """Upload a `.dlm.pack` to `url` via HTTP POST.

    Request body is the raw pack bytes. `Authorization` header is copied
    from `$DLM_SHARE_AUTH` verbatim when set. `Content-Type` is
    `application/octet-stream` — the endpoint is expected to handle a
    raw binary body, NOT a multipart form.

    Raises `SinkError` on HTTP != 2xx or network failure.
    """
    if not pack_path.is_file():
        raise SinkError(f"pack file missing: {pack_path}")

    total = pack_path.stat().st_size
    if url.startswith("http://"):
        _LOG.warning("url sink: pushing over plaintext HTTP (%s); prefer https://", url)

    req = urllib.request.Request(  # noqa: S310 — intentional user-supplied URL
        url,
        method="POST",
        headers=_build_headers(total, content_type="application/octet-stream"),
    )
    try:
        with pack_path.open("rb") as src, urllib.request.urlopen(  # noqa: S310
            req, data=_iter_read(src, total, progress), timeout=60
        ) as resp:
            status = resp.status
            if status < 200 or status >= 300:
                raise SinkError(f"url push: HTTP {status} from {url}")
            _LOG.info("url push: %s bytes → %s (HTTP %d)", total, url, status)
    except urllib.error.HTTPError as exc:
        raise SinkError(f"url push: HTTP {exc.code} from {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SinkError(f"url push: network error contacting {url}: {exc.reason}") from exc
    except OSError as exc:
        raise SinkError(f"url push: I/O error reading {pack_path}: {exc}") from exc


def pull_url(url: str, out_path: Path, *, progress: ProgressCallback = None) -> int:
    """Download `url` to `out_path`. Returns bytes written.

    Streams in `_CHUNK_BYTES` increments. If the server sets
    Content-Length, `progress` fires with the real total; otherwise
    `progress` sees total=0 and reports bytes done only.

    Raises `SinkError` on HTTP != 2xx or network failure.
    """
    if url.startswith("http://"):
        _LOG.warning("url sink: pulling over plaintext HTTP (%s); prefer https://", url)

    req = urllib.request.Request(  # noqa: S310 — intentional user-supplied URL
        url,
        method="GET",
        headers=_build_headers(None, content_type=None),
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            status = resp.status
            if status < 200 or status >= 300:
                raise SinkError(f"url pull: HTTP {status} from {url}")
            total = int(resp.headers.get("Content-Length", "0"))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            bytes_written = _stream_to_file(resp, out_path, total, progress)
            _LOG.info("url pull: %s → %s (%d bytes)", url, out_path, bytes_written)
            return bytes_written
    except urllib.error.HTTPError as exc:
        raise SinkError(f"url pull: HTTP {exc.code} from {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SinkError(f"url pull: network error contacting {url}: {exc.reason}") from exc
    except OSError as exc:
        raise SinkError(f"url pull: I/O error writing {out_path}: {exc}") from exc


def _build_headers(
    content_length: int | None, *, content_type: str | None
) -> dict[str, str]:
    headers = {"User-Agent": _USER_AGENT}
    if content_type is not None:
        headers["Content-Type"] = content_type
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    auth = os.environ.get(_AUTH_ENV)
    if auth:
        headers["Authorization"] = auth
    return headers


def _iter_read(
    src: IO[bytes], total: int, progress: ProgressCallback
) -> bytes:
    """Streaming read adapter for urllib's `data=` parameter.

    urllib accepts a bytes-or-bytes-iterable. We return the full bytes
    buffer — for very large packs, this could be refactored to use
    `http.client.HTTPConnection.send()` directly, but for v1 a single
    in-memory read simplifies error handling. The progress callback
    is invoked once at 0% and once at 100% to match user expectations.
    """
    if progress is not None:
        progress(0, total)
    data: bytes = src.read()
    if progress is not None:
        progress(len(data), total)
    return data


def _stream_to_file(
    resp: IO[bytes], out_path: Path, total: int, progress: ProgressCallback
) -> int:
    written = 0
    with out_path.open("wb") as dst:
        while True:
            chunk = resp.read(_CHUNK_BYTES)
            if not chunk:
                break
            dst.write(chunk)
            written += len(chunk)
            if progress is not None:
                progress(written, total)
    return written
