"""HMAC token issue/verify + expiry + cross-session rejection."""

from __future__ import annotations

import time

import pytest

from dlm.share.errors import PeerAuthError
from dlm.share.peer import (
    RateLimiter,
    RateLimitError,
    ServeOptions,
    new_session,
    resolve_bind,
)


class TestTokenRoundTrip:
    def test_issue_and_verify(self) -> None:
        s = new_session("01HZTEST")
        tok = s.issue_token()
        s.verify_token(tok)  # no raise

    def test_tampered_signature_refused(self) -> None:
        s = new_session("01HZTEST")
        tok = s.issue_token()
        bad = tok[:-3] + "AAA"
        with pytest.raises(PeerAuthError, match="signature mismatch"):
            s.verify_token(bad)

    def test_cross_session_refused(self) -> None:
        s1 = new_session("01HZONE")
        s2 = new_session("01HZTWO")
        tok = s1.issue_token()
        with pytest.raises(PeerAuthError):
            s2.verify_token(tok)

    def test_malformed_base64(self) -> None:
        s = new_session("01HZTEST")
        with pytest.raises(PeerAuthError, match="base64"):
            s.verify_token("!!!not-base64!!!")

    def test_truncated_payload(self) -> None:
        s = new_session("01HZTEST")
        with pytest.raises(PeerAuthError):
            s.verify_token("AAAA")

    def test_expired_token(self) -> None:
        # TTL of 0 — any read after issuance is past-expiry.
        s = new_session("01HZTEST", token_ttl_seconds=0)
        tok = s.issue_token()
        time.sleep(0.01)
        with pytest.raises(PeerAuthError, match="expired"):
            s.verify_token(tok)


class TestBindResolution:
    def test_default_loopback(self) -> None:
        assert resolve_bind(ServeOptions()) == "127.0.0.1"

    def test_public_alone_refused(self, caplog: object) -> None:
        # `--public` without `--i-know-this-is-public` must NOT bind 0.0.0.0.
        import logging

        with caplog.at_level(logging.WARNING, logger="dlm.share.peer"):  # type: ignore[attr-defined]
            bind = resolve_bind(ServeOptions(public=True))
        assert bind == "127.0.0.1"
        # And the user must see the refusal logged.
        messages = " ".join(r.message for r in caplog.records)  # type: ignore[attr-defined]
        assert "refusing" in messages.lower()

    def test_public_with_ack_binds_all(self) -> None:
        bind = resolve_bind(
            ServeOptions(public=True, i_know_this_is_public=True)
        )
        assert bind == "0.0.0.0"


class TestRateLimiter:
    def test_concurrency_cap(self) -> None:
        rl = RateLimiter(max_concurrency=2, rate_limit_per_min=1000)
        rl.check_and_acquire()
        rl.check_and_acquire()
        with pytest.raises(RateLimitError, match="concurrent"):
            rl.check_and_acquire()
        rl.release()
        rl.check_and_acquire()  # slot freed

    def test_rate_cap(self) -> None:
        rl = RateLimiter(max_concurrency=100, rate_limit_per_min=3)
        rl.check_and_acquire()
        rl.check_and_acquire()
        rl.check_and_acquire()
        with pytest.raises(RateLimitError, match="req/min"):
            rl.check_and_acquire()

    def test_release_idempotent_on_zero(self) -> None:
        # Release more than was acquired — shouldn't go negative.
        rl = RateLimiter()
        rl.release()
        rl.release()
        assert rl.active == 0
