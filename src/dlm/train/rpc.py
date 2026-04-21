"""JSON-RPC server for `dlm train --listen-rpc`.

Narrow surface by design: one method (`inject_probe`), one auth pattern
(bearer token from ``DLM_PROBE_TOKEN``), one reply shape. Sway's sink
POSTs newline-delimited JSON; we respond with JSON on stdlib's
`http.server` — no FastAPI / aiohttp dep for what is effectively a
single endpoint.

Contract:

- POST /rpc, `Authorization: Bearer <token>`
- Body: ``{"method": "inject_probe", "params": {"prompt", "reference",
  "tags"}}``
- 200: ``{"accepted": bool, "next_cycle_eta_s": int, "queue_depth": int}``
- 400: malformed payload (parse / schema / missing fields)
- 401: token missing or wrong
- 429: queue past capacity
- 404: unknown method
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any

from dlm.train.inject import InjectedProbe, InjectedProbeQueue, QueueFullError

if TYPE_CHECKING:
    from collections.abc import Callable

_LOGGER = logging.getLogger(__name__)

_RPC_PATH = "/rpc"
_METHOD_INJECT_PROBE = "inject_probe"
_MAX_BODY_BYTES = 64 * 1024  # generous for prompts; bounds DOS surface


def make_handler(
    *,
    queue: InjectedProbeQueue,
    token: str,
    next_cycle_eta_s: Callable[[], int],
) -> type[BaseHTTPRequestHandler]:
    """Build a request-handler class bound to the given queue + token.

    Returned as a class (not instance) because `HTTPServer` instantiates
    a fresh handler per request. Closure captures are the only way to
    hand them the server-state without globals.
    """

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Route stdlib's default stderr prints through our logger.
            _LOGGER.debug("rpc %s", format % args)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != _RPC_PATH:
                self._send_json(404, {"error": f"unknown path {self.path}"})
                return

            auth = self.headers.get("Authorization", "")
            if not _check_bearer(auth, token):
                self._send_json(401, {"error": "missing or invalid bearer token"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                return
            if length <= 0:
                self._send_json(400, {"error": "empty body"})
                return
            if length > _MAX_BODY_BYTES:
                self._send_json(400, {"error": f"body exceeds {_MAX_BODY_BYTES} bytes"})
                return

            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                self._send_json(400, {"error": f"malformed JSON: {exc}"})
                return

            if not isinstance(payload, dict):
                self._send_json(400, {"error": "payload must be a JSON object"})
                return

            method = payload.get("method")
            if method != _METHOD_INJECT_PROBE:
                self._send_json(404, {"error": f"unknown method {method!r}"})
                return

            params = payload.get("params", {})
            if not isinstance(params, dict):
                self._send_json(400, {"error": "`params` must be an object"})
                return

            prompt = params.get("prompt")
            reference = params.get("reference")
            tags = params.get("tags", [])
            if not isinstance(prompt, str) or not prompt.strip():
                self._send_json(400, {"error": "`params.prompt` must be a non-empty string"})
                return
            if not isinstance(reference, str) or not reference.strip():
                self._send_json(400, {"error": "`params.reference` must be a non-empty string"})
                return
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                self._send_json(400, {"error": "`params.tags` must be a list of strings"})
                return

            probe = InjectedProbe(
                prompt=prompt.strip(),
                reference=reference.strip(),
                tags=tuple(tags),
                source_addr=self.client_address[0] if self.client_address else "",
            )
            try:
                queue.enqueue(probe)
            except QueueFullError as exc:
                self._send_json(429, {"error": str(exc), "queue_depth": queue.depth()})
                return

            self._send_json(
                200,
                {
                    "accepted": True,
                    "next_cycle_eta_s": int(next_cycle_eta_s()),
                    "queue_depth": queue.depth(),
                },
            )

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            rendered = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(rendered)))
            self.end_headers()
            self.wfile.write(rendered)

    return _Handler


def _check_bearer(auth_header: str, expected: str) -> bool:
    """Constant-time check of `Authorization: Bearer <token>`."""
    prefix = "Bearer "
    if not auth_header.startswith(prefix):
        return False
    presented = auth_header[len(prefix) :].strip()
    if not presented or not expected:
        return False
    # Constant-time compare: avoid length-leak by padding the shorter.
    a = presented.encode("utf-8")
    b = expected.encode("utf-8")
    if len(a) != len(b):
        return False
    mismatch = 0
    for x, y in zip(a, b, strict=True):
        mismatch |= x ^ y
    return mismatch == 0


class ProbeRpcServer:
    """Threaded HTTP server wrapper — start/stop + queue access."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        token: str,
        queue: InjectedProbeQueue,
        next_cycle_eta_s: Callable[[], int] = lambda: 0,
    ) -> None:
        if not token:
            raise ValueError("bearer token cannot be empty; set DLM_PROBE_TOKEN")
        self._queue = queue
        handler_cls = make_handler(queue=queue, token=token, next_cycle_eta_s=next_cycle_eta_s)
        self._httpd = HTTPServer((host, port), handler_cls)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._httpd.server_address[:2]
        return (str(host), int(port))

    @property
    def queue(self) -> InjectedProbeQueue:
        return self._queue

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("server already started")
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="dlm-probe-rpc", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
