"""Runtime coverage for the peer share transport."""

from __future__ import annotations

import importlib
import socket
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from dlm.share.errors import PeerAuthError, RateLimitError
from dlm.share.peer import (
    RateLimiter,
    ServeHandle,
    ServeOptions,
    _detect_lan_ip,
    _log_connection,
    build_handler,
    new_session,
    pull_peer,
    serve,
)

peer_mod = importlib.import_module("dlm.share.peer")


def _build_test_handler(
    tmp_path: Path,
    *,
    path: str,
) -> tuple[type[object], object, list[tuple[str, str, str, str]], RateLimiter, Path]:
    session = new_session("01HZPEER")
    pack_path = tmp_path / "bundle.dlm.pack"
    pack_path.write_bytes(b"peer-pack")
    rate_limiter = RateLimiter(max_concurrency=4, rate_limit_per_min=30)
    logs: list[tuple[str, str, str, str]] = []

    handler_cls = build_handler(session, pack_path, rate_limiter)
    handler = object.__new__(handler_cls)
    handler.path = path
    handler.client_address = ("127.0.0.1", 7337)
    handler.send_error = lambda code, message: logs.append(("error", str(code), message, ""))  # type: ignore[attr-defined]
    handler._stream_pack = lambda path: logs.append(("stream", str(path), "", ""))  # type: ignore[attr-defined]
    return handler_cls, handler, logs, rate_limiter, pack_path


class TestPeerHandler:
    def test_log_message_is_silent(self, tmp_path: Path) -> None:
        handler_cls, handler, _logs, _rate_limiter, _pack_path = _build_test_handler(
            tmp_path, path="/ignored"
        )
        assert handler_cls.log_message(handler, "%s", "ignored") is None

    def test_handler_rejects_unknown_dlm_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler_cls, handler, events, _rate_limiter, _pack_path = _build_test_handler(
            tmp_path, path="/wrong?token=abc"
        )
        request_logs: list[tuple[str, str, str, str]] = []
        monkeypatch.setattr(
            peer_mod,
            "_log_connection",
            lambda ip, method, path, status: request_logs.append((ip, method, path, status)),
        )

        handler_cls.do_GET(handler)

        assert events == [("error", "404", "unknown dlm_id", "")]
        assert request_logs == [
            ("127.0.0.1", "GET", "/wrong", "start"),
            ("127.0.0.1", "GET", "/wrong", "404 unknown dlm_id"),
        ]

    def test_handler_rejects_missing_token(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler_cls, handler, events, _rate_limiter, _pack_path = _build_test_handler(
            tmp_path, path="/01HZPEER"
        )
        request_logs: list[tuple[str, str, str, str]] = []
        monkeypatch.setattr(
            peer_mod,
            "_log_connection",
            lambda ip, method, path, status: request_logs.append((ip, method, path, status)),
        )

        handler_cls.do_GET(handler)

        assert events == [("error", "401", "missing token", "")]
        assert request_logs == [
            ("127.0.0.1", "GET", "/01HZPEER", "start"),
            ("127.0.0.1", "GET", "/01HZPEER", "401 missing token"),
        ]

    def test_handler_rejects_bad_token(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler_cls, handler, events, _rate_limiter, _pack_path = _build_test_handler(
            tmp_path, path="/01HZPEER?token=bad"
        )
        request_logs: list[tuple[str, str, str, str]] = []
        monkeypatch.setattr(
            peer_mod,
            "_log_connection",
            lambda ip, method, path, status: request_logs.append((ip, method, path, status)),
        )
        monkeypatch.setattr(
            peer_mod.PeerSession,
            "verify_token",
            lambda self, token: (_ for _ in ()).throw(PeerAuthError("bad token")),
        )

        handler_cls.do_GET(handler)

        assert events == [("error", "403", "token rejected", "")]
        assert request_logs == [
            ("127.0.0.1", "GET", "/01HZPEER", "start"),
            ("127.0.0.1", "GET", "/01HZPEER", "403 bad token"),
        ]

    def test_handler_rejects_rate_limited(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler_cls, handler, events, rate_limiter, _pack_path = _build_test_handler(
            tmp_path, path="/01HZPEER?token=good"
        )
        request_logs: list[tuple[str, str, str, str]] = []
        monkeypatch.setattr(
            peer_mod,
            "_log_connection",
            lambda ip, method, path, status: request_logs.append((ip, method, path, status)),
        )
        monkeypatch.setattr(peer_mod.PeerSession, "verify_token", lambda self, token: None)
        monkeypatch.setattr(
            rate_limiter,
            "check_and_acquire",
            lambda: (_ for _ in ()).throw(RateLimitError("too many")),
        )

        handler_cls.do_GET(handler)

        assert events == [("error", "429", "rate limited", "")]
        assert request_logs == [
            ("127.0.0.1", "GET", "/01HZPEER", "start"),
            ("127.0.0.1", "GET", "/01HZPEER", "429 too many"),
        ]

    def test_handler_streams_pack_and_releases_limiter(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler_cls, handler, events, rate_limiter, pack_path = _build_test_handler(
            tmp_path, path="/01HZPEER?token=good"
        )
        request_logs: list[tuple[str, str, str, str]] = []
        monkeypatch.setattr(
            peer_mod,
            "_log_connection",
            lambda ip, method, path, status: request_logs.append((ip, method, path, status)),
        )
        monkeypatch.setattr(peer_mod.PeerSession, "verify_token", lambda self, token: None)

        handler_cls.do_GET(handler)

        assert events == [("stream", str(pack_path), "", "")]
        assert rate_limiter.active == 0
        assert request_logs == [
            ("127.0.0.1", "GET", "/01HZPEER", "start"),
            ("127.0.0.1", "GET", "/01HZPEER", "200 complete"),
        ]

    def test_stream_pack_writes_headers_and_body(self, tmp_path: Path) -> None:
        handler_cls, handler, _events, _rate_limiter, pack_path = _build_test_handler(
            tmp_path, path="/ignored"
        )
        responses: list[tuple[str, str]] = []
        body = BytesIO()
        handler.wfile = body
        handler.send_response = lambda status: responses.append(("status", str(status)))  # type: ignore[attr-defined]
        handler.send_header = lambda name, value: responses.append((name, value))  # type: ignore[attr-defined]
        handler.end_headers = lambda: responses.append(("end", ""))  # type: ignore[attr-defined]

        handler_cls._stream_pack(handler, pack_path)

        assert responses == [
            ("status", "200"),
            ("Content-Type", "application/octet-stream"),
            ("Content-Length", str(len(b"peer-pack"))),
            ("end", ""),
        ]
        assert body.getvalue() == b"peer-pack"


class TestPeerHelpers:
    def test_log_connection_emits_metadata_only(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("INFO")

        _log_connection("127.0.0.1", "GET", "/01HZPEER", "200 complete")

        assert "peer: GET /01HZPEER 200 complete from 127.0.0.1" in caplog.text

    def test_pull_peer_reuses_url_sink(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.share.url_sink as url_sink

        out_path = tmp_path / "incoming.dlm.pack"
        seen: dict[str, object] = {}

        def _fake_pull_url(url: str, actual_out: Path, *, progress: object | None = None) -> int:
            seen["url"] = url
            seen["out"] = actual_out
            seen["progress"] = progress
            return 42

        monkeypatch.setattr(url_sink, "pull_url", _fake_pull_url)

        result = pull_peer("host:7337/01HZPEER?token=abc", out_path, progress=None)

        assert result == 42
        assert seen == {
            "url": "http://host:7337/01HZPEER?token=abc",
            "out": out_path,
            "progress": None,
        }


class TestServeHandle:
    def test_peer_url_uses_bind_host_for_loopback(self) -> None:
        handle = ServeHandle(
            session=SimpleNamespace(dlm_id="01HZPEER"),
            bind_host="127.0.0.1",
            port=7337,
            token="abc",
            _server=SimpleNamespace(),
        )

        assert handle.peer_url == "peer://127.0.0.1:7337/01HZPEER?token=abc"

    def test_peer_url_detects_lan_ip_for_public_bind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handle = ServeHandle(
            session=SimpleNamespace(dlm_id="01HZPEER"),
            bind_host="0.0.0.0",
            port=7337,
            token="abc",
            _server=SimpleNamespace(),
        )
        monkeypatch.setattr(peer_mod, "_detect_lan_ip", lambda: "192.168.1.9")

        assert handle.peer_url == "peer://192.168.1.9:7337/01HZPEER?token=abc"

    def test_wait_shutdown_stops_server_cleanly(self) -> None:
        calls: list[str] = []
        server = SimpleNamespace(
            serve_forever=lambda: calls.append("serve_forever"),
            shutdown=lambda: calls.append("shutdown"),
            server_close=lambda: calls.append("server_close"),
        )
        handle = ServeHandle(
            session=SimpleNamespace(dlm_id="01HZPEER"),
            bind_host="127.0.0.1",
            port=7337,
            token="abc",
            _server=server,
        )

        handle.wait_shutdown()

        assert calls == ["serve_forever", "shutdown", "server_close"]

    def test_wait_shutdown_handles_keyboard_interrupt(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        calls: list[str] = []

        def _serve_forever() -> None:
            calls.append("serve_forever")
            raise KeyboardInterrupt

        server = SimpleNamespace(
            serve_forever=_serve_forever,
            shutdown=lambda: calls.append("shutdown"),
            server_close=lambda: calls.append("server_close"),
        )
        handle = ServeHandle(
            session=SimpleNamespace(dlm_id="01HZPEER"),
            bind_host="127.0.0.1",
            port=7337,
            token="abc",
            _server=server,
        )
        caplog.set_level("INFO")

        handle.wait_shutdown()

        assert calls == ["serve_forever", "shutdown", "server_close"]
        assert "shutdown requested" in caplog.text


class TestServe:
    def test_serve_builds_handle(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pack_path = tmp_path / "bundle.dlm.pack"
        pack_path.write_bytes(b"peer-pack")
        handler_cls = type("FakeHandler", (), {})
        server_calls: dict[str, object] = {}

        class FakeSession:
            dlm_id = "01HZPEER"

            def issue_token(self) -> str:
                return "issued-token"

        class FakeServer:
            def __init__(self, address: tuple[str, int], handler: type[object]) -> None:
                server_calls["address"] = address
                server_calls["handler"] = handler

        monkeypatch.setattr(
            peer_mod, "new_session", lambda dlm_id, token_ttl_seconds: FakeSession()
        )
        monkeypatch.setattr(
            peer_mod, "build_handler", lambda session, actual_pack, limiter: handler_cls
        )
        monkeypatch.setattr(peer_mod, "resolve_bind", lambda opts: "127.0.0.1")
        monkeypatch.setattr(peer_mod.http.server, "ThreadingHTTPServer", FakeServer)

        handle = serve("01HZPEER", pack_path, ServeOptions(port=8123))

        assert handle.session.dlm_id == "01HZPEER"
        assert handle.bind_host == "127.0.0.1"
        assert handle.port == 8123
        assert handle.token == "issued-token"
        assert server_calls == {
            "address": ("127.0.0.1", 8123),
            "handler": handler_cls,
        }


class TestDetectLanIp:
    def test_detect_lan_ip_returns_socket_address(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeSocket:
            def settimeout(self, value: float) -> None:
                assert value == 0.1

            def connect(self, target: tuple[str, int]) -> None:
                assert target == ("10.254.254.254", 1)

            def getsockname(self) -> tuple[str, int]:
                return ("192.168.1.7", 9999)

            def __enter__(self) -> FakeSocket:
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                return None

        monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: FakeSocket())

        assert _detect_lan_ip() == "192.168.1.7"

    def test_detect_lan_ip_returns_placeholder_on_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeSocket:
            def __enter__(self) -> FakeSocket:
                raise OSError("no route")

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                return None

        monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: FakeSocket())

        assert _detect_lan_ip() == "<lan-ip>"
