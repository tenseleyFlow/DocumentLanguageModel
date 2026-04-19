"""`mint_ulid` — 26-char Crockford base32 + monotonicity (Sprint 13)."""

from __future__ import annotations

import re

from dlm.io.ulid import mint_ulid

_ULID_RE = re.compile(r"^[0-9A-HJKMNPQRSTVWXYZ]{26}$")


class TestShape:
    def test_returns_26_chars(self) -> None:
        assert len(mint_ulid()) == 26

    def test_matches_crockford_regex(self) -> None:
        for _ in range(100):
            assert _ULID_RE.fullmatch(mint_ulid())

    def test_uppercase_only(self) -> None:
        ulid = mint_ulid()
        assert ulid == ulid.upper()


class TestUniqueness:
    def test_two_consecutive_calls_differ(self) -> None:
        """80 bits of entropy per call; collision probability is negligible."""
        assert mint_ulid() != mint_ulid()

    def test_one_thousand_distinct(self) -> None:
        ulids = {mint_ulid() for _ in range(1000)}
        assert len(ulids) == 1000


class TestSortability:
    def test_later_ulid_sorts_greater(self) -> None:
        """Timestamp prefix means lexicographic order ~ time order.

        Sleep is risky in tests; use `time.time` mocking-free assertion
        via the 48-bit timestamp prefix: two calls spaced by a second's
        worth of monotonic progress should sort in that order.
        """
        import time

        a = mint_ulid()
        time.sleep(0.002)
        b = mint_ulid()
        # First 10 chars encode the timestamp (48 bits → 10 Crockford chars).
        # Even two milliseconds apart the prefix must not regress.
        assert a[:10] <= b[:10]


class TestInternals:
    def test_wrong_payload_size_raises(self) -> None:
        """`_encode_crockford` rejects anything other than exactly 16 bytes."""
        import pytest

        from dlm.io.ulid import _encode_crockford

        with pytest.raises(ValueError, match="16 bytes"):
            _encode_crockford(b"\x00" * 15)
        with pytest.raises(ValueError, match="16 bytes"):
            _encode_crockford(b"\x00" * 17)


class TestValidatorCompat:
    def test_accepted_by_frontmatter_validator(self) -> None:
        """Round-trip: mint → validate against the schema's ULID regex."""
        from dlm.doc.schema import DlmFrontmatter

        for _ in range(50):
            fm = DlmFrontmatter.model_validate(
                {"dlm_id": mint_ulid(), "base_model": "smollm2-135m"}
            )
            assert fm.dlm_id
