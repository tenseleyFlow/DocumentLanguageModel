"""Unit tests for the probe-RPC server.

Exercises the handler directly via `urllib` against a real bound port
rather than mocking `BaseHTTPRequestHandler` internals — the surface is
narrow enough that actual-socket tests stay fast and have no flake
potential beyond "picked a used port," mitigated by letting the OS
assign (`port=0`).
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from dlm.train.inject import InjectedProbeQueue
from dlm.train.rpc import ProbeRpcServer, _check_bearer

_TOKEN = "test-token-123"


@pytest.fixture
def server() -> Iterator[ProbeRpcServer]:
    queue = InjectedProbeQueue(capacity=4)
    try:
        srv = ProbeRpcServer(
            host="127.0.0.1", port=0, token=_TOKEN, queue=queue, next_cycle_eta_s=lambda: 42
        )
    except PermissionError as exc:
        pytest.skip(f"loopback bind blocked on this host: {exc}")
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def _post(
    server: ProbeRpcServer,
    *,
    body: dict[str, Any] | str,
    token: str | None = _TOKEN,
    path: str = "/rpc",
) -> tuple[int, dict[str, Any]]:
    host, port = server.address
    url = f"http://{host}:{port}{path}"
    raw = body if isinstance(body, str) else json.dumps(body)
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=raw.encode("utf-8"), headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=5.0)  # noqa: S310
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())
    return resp.status, json.loads(resp.read())


def _raw_post(
    server: ProbeRpcServer,
    *,
    headers: dict[str, str],
    body: bytes = b"",
    path: str = "/rpc",
) -> tuple[int, dict[str, Any]]:
    host, port = server.address
    lines = [
        f"POST {path} HTTP/1.1",
        f"Host: {host}:{port}",
        *[f"{key}: {value}" for key, value in headers.items()],
        "",
        "",
    ]
    request = "\r\n".join(lines).encode("utf-8") + body
    with socket.create_connection((host, port), timeout=5.0) as sock:
        sock.sendall(request)
        response = b""
        while b"\r\n\r\n" not in response:
            response += sock.recv(4096)
        head, rest = response.split(b"\r\n\r\n", 1)
        header_lines = head.decode("iso-8859-1").split("\r\n")
        status = int(header_lines[0].split()[1])
        parsed_headers: dict[str, str] = {}
        for line in header_lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed_headers[key.lower()] = value.strip()
        content_length = int(parsed_headers.get("content-length", "0"))
        while len(rest) < content_length:
            rest += sock.recv(4096)
    return status, json.loads(rest[:content_length].decode("utf-8"))


class TestHappyPath:
    def test_inject_probe_accepted(self, server: ProbeRpcServer) -> None:
        status, body = _post(
            server,
            body={
                "method": "inject_probe",
                "params": {"prompt": "what is X?", "reference": "Y.", "tags": ["sway"]},
            },
        )
        assert status == 200
        assert body == {"accepted": True, "next_cycle_eta_s": 42, "queue_depth": 1}
        drained = server.queue.drain()
        assert len(drained) == 1
        assert drained[0].prompt == "what is X?"
        assert drained[0].tags == ("sway",)


class TestAuth:
    def test_missing_token_401(self, server: ProbeRpcServer) -> None:
        status, body = _post(
            server,
            body={"method": "inject_probe", "params": {"prompt": "q", "reference": "a"}},
            token=None,
        )
        assert status == 401
        assert "bearer" in body["error"].lower()

    def test_wrong_token_401(self, server: ProbeRpcServer) -> None:
        status, body = _post(
            server,
            body={"method": "inject_probe", "params": {"prompt": "q", "reference": "a"}},
            token="wrong-token",
        )
        assert status == 401


class TestMalformedPayload:
    def test_bad_json_400(self, server: ProbeRpcServer) -> None:
        status, body = _post(server, body="not json {")
        assert status == 400
        assert "malformed" in body["error"].lower()

    def test_invalid_content_length_400(self, server: ProbeRpcServer) -> None:
        status, body = _raw_post(
            server,
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                "Content-Type": "application/json",
                "Content-Length": "nope",
            },
        )
        assert status == 400
        assert "content-length" in body["error"].lower()

    def test_empty_body_400(self, server: ProbeRpcServer) -> None:
        status, body = _raw_post(
            server,
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                "Content-Type": "application/json",
                "Content-Length": "0",
            },
        )
        assert status == 400
        assert "empty body" in body["error"].lower()

    def test_oversized_body_400(self, server: ProbeRpcServer) -> None:
        status, body = _raw_post(
            server,
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                "Content-Type": "application/json",
                "Content-Length": str(70 * 1024),
            },
        )
        assert status == 400
        assert "exceeds" in body["error"].lower()

    def test_payload_must_be_object(self, server: ProbeRpcServer) -> None:
        status, body = _post(server, body="[]")
        assert status == 400
        assert "json object" in body["error"].lower()

    def test_missing_prompt_400(self, server: ProbeRpcServer) -> None:
        status, body = _post(server, body={"method": "inject_probe", "params": {"reference": "a"}})
        assert status == 400
        assert "prompt" in body["error"].lower()

    def test_params_must_be_object(self, server: ProbeRpcServer) -> None:
        status, body = _post(server, body={"method": "inject_probe", "params": "bad"})
        assert status == 400
        assert "`params`" in body["error"]

    def test_empty_reference_400(self, server: ProbeRpcServer) -> None:
        status, body = _post(
            server,
            body={"method": "inject_probe", "params": {"prompt": "q", "reference": "   "}},
        )
        assert status == 400
        assert "reference" in body["error"].lower()

    def test_non_string_tags_400(self, server: ProbeRpcServer) -> None:
        status, body = _post(
            server,
            body={
                "method": "inject_probe",
                "params": {"prompt": "q", "reference": "a", "tags": [1, 2]},
            },
        )
        assert status == 400
        assert "tags" in body["error"].lower()


class TestMethodDispatch:
    def test_unknown_method_404(self, server: ProbeRpcServer) -> None:
        status, body = _post(server, body={"method": "explode", "params": {}})
        assert status == 404
        assert "explode" in body["error"]

    def test_unknown_path_404(self, server: ProbeRpcServer) -> None:
        status, body = _post(server, body={"method": "inject_probe"}, path="/other")
        assert status == 404


class TestCapacity:
    def test_full_queue_429(self, server: ProbeRpcServer) -> None:
        payload = {
            "method": "inject_probe",
            "params": {"prompt": "q", "reference": "a"},
        }
        for _ in range(4):
            status, _ = _post(server, body=payload)
            assert status == 200
        status, body = _post(server, body=payload)
        assert status == 429
        assert body["queue_depth"] == 4


class TestAuthHelper:
    def test_correct_token_matches(self) -> None:
        assert _check_bearer("Bearer abc", "abc") is True

    def test_wrong_token_fails(self) -> None:
        assert _check_bearer("Bearer xyz", "abc") is False

    def test_missing_prefix_fails(self) -> None:
        assert _check_bearer("abc", "abc") is False

    def test_empty_presented_fails(self) -> None:
        assert _check_bearer("Bearer ", "abc") is False

    def test_length_mismatch_fails(self) -> None:
        assert _check_bearer("Bearer abcd", "abc") is False


class TestConstruction:
    def test_empty_token_rejected(self) -> None:
        with pytest.raises(ValueError, match="bearer token"):
            ProbeRpcServer(host="127.0.0.1", port=0, token="", queue=InjectedProbeQueue())

    def test_start_twice_rejected(self) -> None:
        try:
            srv = ProbeRpcServer(
                host="127.0.0.1",
                port=0,
                token=_TOKEN,
                queue=InjectedProbeQueue(),
            )
        except PermissionError as exc:
            pytest.skip(f"loopback bind blocked on this host: {exc}")
        srv.start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                srv.start()
        finally:
            srv.stop()
