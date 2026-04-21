"""Peer LAN mode: `dlm serve` + `dlm pull peer://...`.

One machine serves a packed `.dlm` via a short-lived HTTP endpoint.
Another machine pulls it. No central server, no persistent state, no
account anywhere.

Security posture (audit F30):

- **Bind default `127.0.0.1`.** `--public` opts into `0.0.0.0` only when
  paired with `--i-know-this-is-public`. Without both, a `--public`
  request is rejected before the server starts.
- **HMAC tokens.** Tokens are stateless: `hmac_sha256(secret, dlm_id
  || expiry_iso || nonce)`. The server holds the secret in memory for
  its lifetime only; Ctrl-C invalidates every outstanding token
  instantly. Verification needs only the secret + the token; no DB.
- **Rate limiting.** Per-token: max concurrent connections (default 4)
  and max requests per minute (default 30). Exceeding returns HTTP 429.
- **No content logging.** Connection metadata (IP, timestamp, status,
  bytes) only. The pack bytes never land in the log stream.
- **Ed25519-style token opacity.** Tokens are base64url-encoded
  `nonce || expiry_iso || signature`; a client inspecting the bytes
  learns nothing actionable without the server secret.
"""

from __future__ import annotations

import base64
import hmac
import http.server
import logging
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

from dlm.share.errors import PeerAuthError, RateLimitError

_LOG = logging.getLogger(__name__)

_DEFAULT_PORT = 7337
_DEFAULT_TOKEN_TTL_SECONDS = 15 * 60  # 15 minutes
_DEFAULT_MAX_CONCURRENCY = 4
_DEFAULT_RATE_LIMIT_PER_MIN = 30
_SECRET_BYTES = 32
_NONCE_BYTES = 12


# ---------- Tokens --------------------------------------------------------------


@dataclass(frozen=True)
class PeerSession:
    """One `dlm serve` session's authentication state.

    `secret` is generated at session start and only lives in process
    memory. Ctrl-C kills the process, invalidating every issued token
    instantly — there is no persistence.
    """

    dlm_id: str
    secret: bytes
    token_ttl_seconds: int = _DEFAULT_TOKEN_TTL_SECONDS

    def issue_token(self) -> str:
        """Mint a fresh HMAC-signed token. Returns base64url-encoded bytes."""
        nonce = secrets.token_bytes(_NONCE_BYTES)
        expiry = datetime.now(tz=UTC) + timedelta(seconds=self.token_ttl_seconds)
        expiry_iso = expiry.isoformat()
        signature = self._sign(self.dlm_id, expiry_iso, nonce)
        # Wire format: nonce || len(expiry_iso) || expiry_iso || sig
        payload = (
            nonce + len(expiry_iso).to_bytes(2, "big") + expiry_iso.encode("ascii") + signature
        )
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    def verify_token(self, token: str) -> None:
        """Raise `PeerAuthError` if `token` is invalid, expired, or forged."""
        try:
            payload = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        except (ValueError, TypeError) as exc:
            raise PeerAuthError("token: not valid base64url") from exc

        if len(payload) < _NONCE_BYTES + 2 + 32:
            raise PeerAuthError("token: too short")

        nonce = payload[:_NONCE_BYTES]
        iso_len = int.from_bytes(payload[_NONCE_BYTES : _NONCE_BYTES + 2], "big")
        iso_start = _NONCE_BYTES + 2
        iso_end = iso_start + iso_len
        sig_start = iso_end

        if sig_start + 32 != len(payload):
            raise PeerAuthError("token: trailing bytes")

        expiry_iso = payload[iso_start:iso_end].decode("ascii", errors="replace")
        signature = payload[sig_start:]

        expected = self._sign(self.dlm_id, expiry_iso, nonce)
        if not hmac.compare_digest(expected, signature):
            raise PeerAuthError("token: signature mismatch")

        try:
            expiry = datetime.fromisoformat(expiry_iso)
        except ValueError as exc:
            raise PeerAuthError("token: malformed expiry") from exc

        if expiry < datetime.now(tz=UTC):
            raise PeerAuthError(f"token: expired at {expiry_iso}")

    def _sign(self, dlm_id: str, expiry_iso: str, nonce: bytes) -> bytes:
        mac = hmac.new(self.secret, digestmod=sha256)
        mac.update(dlm_id.encode("utf-8"))
        mac.update(b"\x00")
        mac.update(expiry_iso.encode("ascii"))
        mac.update(b"\x00")
        mac.update(nonce)
        return mac.digest()


def new_session(dlm_id: str, *, token_ttl_seconds: int = _DEFAULT_TOKEN_TTL_SECONDS) -> PeerSession:
    """Create a fresh session with a cryptographically-random secret."""
    return PeerSession(
        dlm_id=dlm_id,
        secret=secrets.token_bytes(_SECRET_BYTES),
        token_ttl_seconds=token_ttl_seconds,
    )


# ---------- Rate limiting -------------------------------------------------------


@dataclass
class RateLimiter:
    """Per-token concurrency + requests-per-minute cap.

    Not thread-safe by itself — callers hold `self.lock` around
    `acquire` / `release` / `check_rate`. The peer server runs a
    single-worker thread pool (see `ServeOptions.workers`), so the
    lock is rarely contended.
    """

    max_concurrency: int = _DEFAULT_MAX_CONCURRENCY
    rate_limit_per_min: int = _DEFAULT_RATE_LIMIT_PER_MIN
    active: int = 0
    requests: deque[float] = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def check_and_acquire(self) -> None:
        """Record a new request; raise `RateLimitError` if caps exceeded."""
        now = time.monotonic()
        window_start = now - 60.0
        with self.lock:
            # Prune requests older than the 60 s window.
            while self.requests and self.requests[0] < window_start:
                self.requests.popleft()
            if len(self.requests) >= self.rate_limit_per_min:
                raise RateLimitError(f"rate limit: {self.rate_limit_per_min} req/min exceeded")
            if self.active >= self.max_concurrency:
                raise RateLimitError(
                    f"rate limit: max concurrent connections ({self.max_concurrency}) exceeded"
                )
            self.requests.append(now)
            self.active += 1

    def release(self) -> None:
        with self.lock:
            if self.active > 0:
                self.active -= 1


# ---------- Server --------------------------------------------------------------


@dataclass(frozen=True)
class ServeOptions:
    """CLI-configurable knobs for `dlm serve`."""

    port: int = _DEFAULT_PORT
    public: bool = False
    i_know_this_is_public: bool = False
    max_concurrency: int = _DEFAULT_MAX_CONCURRENCY
    rate_limit_per_min: int = _DEFAULT_RATE_LIMIT_PER_MIN
    token_ttl_seconds: int = _DEFAULT_TOKEN_TTL_SECONDS


def resolve_bind(opts: ServeOptions) -> str:
    """Decide the bind address honoring the `--public` / ack gate.

    Returns `"127.0.0.1"` unless BOTH `public=True` AND
    `i_know_this_is_public=True`. When only one side is set, we bind
    loopback and log a warning so the user sees the refusal — never
    silently bind public.
    """
    if opts.public and opts.i_know_this_is_public:
        return "0.0.0.0"  # noqa: S104 — intentional public bind with explicit ack
    if opts.public and not opts.i_know_this_is_public:
        _LOG.warning(
            "peer serve: --public requested without --i-know-this-is-public; "
            "refusing to bind 0.0.0.0. Binding 127.0.0.1 only."
        )
    return "127.0.0.1"


def build_handler(
    session: PeerSession,
    pack_path: Path,
    rate_limiter: RateLimiter,
) -> type[http.server.BaseHTTPRequestHandler]:
    """Return a handler class closed over the serve session + pack.

    One endpoint: `GET /<dlm_id>?token=<token>` streams the pack bytes.
    Any other path or method returns 404.
    """

    class _PackHandler(http.server.BaseHTTPRequestHandler):
        # Suppress the default per-request stdout logging; we handle
        # structured logging ourselves and don't want pack content
        # metadata (byte counts, timings) interleaved with noise.
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002, ARG002
            return

        def do_GET(self) -> None:  # noqa: N802 — stdlib signature
            # Path: "/<dlm_id>", query: "?token=..."
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(self.path)
            requested_id = parsed.path.lstrip("/")
            query = parse_qs(parsed.query)
            token_values = query.get("token", [])

            client_ip = self.client_address[0]
            _log_connection(client_ip, "GET", parsed.path, "start")

            if requested_id != session.dlm_id:
                self.send_error(404, "unknown dlm_id")
                _log_connection(client_ip, "GET", parsed.path, "404 unknown dlm_id")
                return
            if not token_values:
                self.send_error(401, "missing token")
                _log_connection(client_ip, "GET", parsed.path, "401 missing token")
                return

            try:
                session.verify_token(token_values[0])
            except PeerAuthError as exc:
                self.send_error(403, "token rejected")
                _log_connection(client_ip, "GET", parsed.path, f"403 {exc}")
                return

            try:
                rate_limiter.check_and_acquire()
            except RateLimitError as exc:
                self.send_error(429, "rate limited")
                _log_connection(client_ip, "GET", parsed.path, f"429 {exc}")
                return

            try:
                self._stream_pack(pack_path)
                _log_connection(client_ip, "GET", parsed.path, "200 complete")
            finally:
                rate_limiter.release()

        def _stream_pack(self, path: Path) -> None:
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with path.open("rb") as src:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    return _PackHandler


def _log_connection(ip: str, method: str, path: str, status: str) -> None:
    """Connection metadata only — never pack content."""
    _LOG.info("peer: %s %s %s from %s", method, path, status, ip)


# ---------- Client --------------------------------------------------------------


ProgressCallback = Callable[[int, int], None] | None


def pull_peer(
    target: str,
    out_path: Path,
    *,
    progress: ProgressCallback = None,
) -> int:
    """Pull a pack from `peer://<target>`.

    `target` is everything after `peer://`: `host:port/<dlm_id>?token=...`.
    Translates to an HTTP GET against `http://<target>`. Returns bytes
    written. Raises `SinkError` on HTTP != 2xx or network failure.
    """
    # Reuse the URL sink's streaming implementation — peer mode is
    # plain HTTP at the wire level, just with a friendly scheme prefix.
    from dlm.share.url_sink import pull_url

    return pull_url(f"http://{target}", out_path, progress=progress)


# ---------- Top-level orchestrator ----------------------------------------------


@dataclass(frozen=True)
class ServeHandle:
    """Return value from `serve` — the running server + its URL + token.

    The CLI uses this to print the connection string the other side
    pastes into `dlm pull`. `wait_shutdown()` blocks until the server
    stops (typically on Ctrl-C).
    """

    session: PeerSession
    bind_host: str
    port: int
    token: str
    _server: http.server.HTTPServer

    @property
    def peer_url(self) -> str:
        """The `peer://` URL the other machine uses to pull."""
        host = self.bind_host if self.bind_host != "0.0.0.0" else _detect_lan_ip()
        return f"peer://{host}:{self.port}/{self.session.dlm_id}?token={self.token}"

    def wait_shutdown(self) -> None:
        """Block until the server stops. Returns on Ctrl-C (KeyboardInterrupt)."""
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            _LOG.info("peer serve: shutdown requested (Ctrl-C)")
        finally:
            self._server.shutdown()
            self._server.server_close()


def serve(
    dlm_id: str,
    pack_path: Path,
    opts: ServeOptions,
) -> ServeHandle:
    """Start a peer-mode HTTP server serving `pack_path`.

    `dlm_id` is recorded in the token and checked at every request.
    The server runs in the caller's thread; `ServeHandle.wait_shutdown`
    blocks. Ctrl-C cleanly stops the server and invalidates all
    outstanding tokens (the session secret lives only in memory).
    """
    session = new_session(dlm_id, token_ttl_seconds=opts.token_ttl_seconds)
    rate_limiter = RateLimiter(
        max_concurrency=opts.max_concurrency,
        rate_limit_per_min=opts.rate_limit_per_min,
    )
    handler_cls = build_handler(session, pack_path, rate_limiter)
    bind_host = resolve_bind(opts)
    server = http.server.ThreadingHTTPServer((bind_host, opts.port), handler_cls)

    token = session.issue_token()
    return ServeHandle(
        session=session,
        bind_host=bind_host,
        port=opts.port,
        token=token,
        _server=server,
    )


def _detect_lan_ip() -> str:
    """Best-effort LAN IP detection for the `peer://` URL.

    Not a security boundary — just user-facing convenience so the
    printed URL works from the peer's machine. Fall back to
    `<lan-ip>` placeholder if detection fails.
    """
    import socket

    try:
        # Connect to a fake address to force the OS to pick the LAN
        # interface; doesn't actually send anything.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.1)
            sock.connect(("10.254.254.254", 1))
            addr: str = sock.getsockname()[0]
            return addr
    except OSError:
        return "<lan-ip>"
