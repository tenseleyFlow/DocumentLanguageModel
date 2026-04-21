"""Signing wrapper: minisign availability probe + verify-result shape."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dlm.share.signing import (
    MinisignNotAvailableError,
    VerifyResult,
    VerifyStatus,
    minisign_available,
    sign_file,
    verify_signature,
)


class TestAvailability:
    def test_available_probe_returns_bool(self) -> None:
        # Just check it doesn't raise and returns a bool — the system's
        # actual state varies per runner.
        assert isinstance(minisign_available(), bool)


class TestVerifyMissingSignature:
    def test_unsigned_when_sig_absent(self, tmp_path: Path) -> None:
        target = tmp_path / "pack.bin"
        target.write_bytes(b"payload")
        sig = tmp_path / "pack.bin.minisig"
        # sig file doesn't exist
        result = verify_signature(target, sig)
        assert result.status == VerifyStatus.UNSIGNED
        assert result.key_path is None


class TestVerifyNoMinisignBinary:
    def test_unverified_when_binary_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "pack.bin"
        target.write_bytes(b"payload")
        sig = tmp_path / "pack.bin.minisig"
        sig.write_bytes(b"fake-sig")
        keys = tmp_path / "trusted-keys"
        keys.mkdir()

        with patch("dlm.share.signing.minisign_available", return_value=False):
            result = verify_signature(target, sig, trusted_keys_dir=keys)
        assert result.status == VerifyStatus.UNVERIFIED
        assert "minisign" in result.detail


class TestVerifyNoTrustedKeys:
    def test_unverified_when_keys_dir_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "pack.bin"
        target.write_bytes(b"payload")
        sig = tmp_path / "pack.bin.minisig"
        sig.write_bytes(b"fake-sig")
        keys = tmp_path / "does-not-exist"

        with patch("dlm.share.signing.minisign_available", return_value=True):
            result = verify_signature(target, sig, trusted_keys_dir=keys)
        assert result.status == VerifyStatus.UNVERIFIED
        assert "trusted keys" in result.detail


class TestVerifyResult:
    def test_constructs_cleanly(self) -> None:
        r = VerifyResult(status=VerifyStatus.VERIFIED, key_path=Path("/tmp/k.pub"))
        assert r.status == VerifyStatus.VERIFIED
        assert r.key_path == Path("/tmp/k.pub")
        assert r.detail == ""


class TestSignRefusesWithoutBinary:
    def test_raises_when_minisign_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "pack.bin"
        target.write_bytes(b"payload")
        with (
            patch("dlm.share.signing.minisign_available", return_value=False),
            pytest.raises(MinisignNotAvailableError, match="not installed"),
        ):
            sign_file(target)
