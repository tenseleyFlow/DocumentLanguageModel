"""End-to-end peer-mode: start a server in a thread, pull from the client.

Avoids subprocess gymnastics — everything runs in one process. The
server runs in a background thread; the test thread is the peer
client. Verifies:

1. The HTTP server accepts a valid token on GET /<dlm_id>
2. The pack bytes arrive intact (byte-for-byte equal)
3. An expired token is refused with HTTP 403
4. An unknown dlm_id is refused with HTTP 404
5. Missing token is refused with HTTP 401
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from dlm.share import ServeHandle


def _start_server_in_thread(tmp_path: Path, *, ttl: int = 600) -> tuple[ServeHandle, threading.Thread, bytes]:
    """Helper: pack a trivial file + start the peer server in a thread.

    Returns `(handle, thread, pack_bytes)`. Caller stops via
    `handle._server.shutdown()` + `thread.join()`.
    """
    from dlm.share import ServeOptions, serve

    # Simulate a "pack" — any bytes will do for the transport test.
    pack = tmp_path / "fake.dlm.pack"
    pack_bytes = b"dlm-pack-contents-" * 256  # ~4 KB
    pack.write_bytes(pack_bytes)

    opts = ServeOptions(port=0, token_ttl_seconds=ttl)  # port=0 → OS picks free port
    try:
        handle = serve("01HZTESTID", pack, opts)
    except PermissionError as exc:
        pytest.skip(f"loopback bind blocked on this host: {exc}")

    thread = threading.Thread(target=handle._server.serve_forever, daemon=True)
    thread.start()

    # Give the server a moment to bind.
    time.sleep(0.05)
    return handle, thread, pack_bytes


def _stop_server(handle: ServeHandle, thread: threading.Thread) -> None:
    handle._server.shutdown()
    handle._server.server_close()
    thread.join(timeout=2.0)


class TestPeerRoundTrip:
    def test_happy_path(self, tmp_path: Path) -> None:
        handle, thread, original = _start_server_in_thread(tmp_path)
        try:
            # Construct the actual bind URL from the handle — resolve_bind
            # returns 127.0.0.1 by default, and port 0 was replaced with
            # the real port by the OS on serve_forever.
            real_port = handle._server.server_address[1]
            url = f"http://127.0.0.1:{real_port}/{handle.session.dlm_id}?token={handle.token}"

            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                assert resp.status == 200
                received = resp.read()
            assert received == original
        finally:
            _stop_server(handle, thread)

    def test_missing_token_refused(self, tmp_path: Path) -> None:
        handle, thread, _ = _start_server_in_thread(tmp_path)
        try:
            real_port = handle._server.server_address[1]
            url = f"http://127.0.0.1:{real_port}/{handle.session.dlm_id}"
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(url, timeout=2)  # noqa: S310
            assert exc_info.value.code == 401
        finally:
            _stop_server(handle, thread)

    def test_bad_token_refused(self, tmp_path: Path) -> None:
        handle, thread, _ = _start_server_in_thread(tmp_path)
        try:
            real_port = handle._server.server_address[1]
            url = f"http://127.0.0.1:{real_port}/{handle.session.dlm_id}?token=garbage"
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(url, timeout=2)  # noqa: S310
            assert exc_info.value.code == 403
        finally:
            _stop_server(handle, thread)

    def test_unknown_dlm_id_refused(self, tmp_path: Path) -> None:
        handle, thread, _ = _start_server_in_thread(tmp_path)
        try:
            real_port = handle._server.server_address[1]
            url = f"http://127.0.0.1:{real_port}/01HZDIFFERENT?token={handle.token}"
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(url, timeout=2)  # noqa: S310
            assert exc_info.value.code == 404
        finally:
            _stop_server(handle, thread)

    def test_expired_token_refused(self, tmp_path: Path) -> None:
        # TTL of 0 → token is born expired.
        handle, thread, _ = _start_server_in_thread(tmp_path, ttl=0)
        try:
            time.sleep(0.01)  # ensure clock moves past expiry
            real_port = handle._server.server_address[1]
            url = f"http://127.0.0.1:{real_port}/{handle.session.dlm_id}?token={handle.token}"
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(url, timeout=2)  # noqa: S310
            assert exc_info.value.code == 403
        finally:
            _stop_server(handle, thread)
