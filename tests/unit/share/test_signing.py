"""Signing wrapper: minisign availability probe + verify-result shape."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dlm.share.errors import ShareError
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


class TestSignFile:
    def test_missing_secret_key_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "pack.bin"
        target.write_bytes(b"payload")

        with (
            patch("dlm.share.signing.minisign_available", return_value=True),
            pytest.raises(Exception, match="secret key not found"),
        ):
            sign_file(target, secret_key=tmp_path / "missing.key")

    def test_nonzero_exit_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "pack.bin"
        target.write_bytes(b"payload")
        secret = tmp_path / "secret.key"
        secret.write_text("key", encoding="utf-8")

        class Result:
            returncode = 7

        with (
            patch("dlm.share.signing.minisign_available", return_value=True),
            patch("subprocess.run", return_value=Result()),
            pytest.raises(Exception, match="exit 7"),
        ):
            sign_file(target, secret_key=secret, comment="demo")

    def test_missing_signature_sidecar_after_success_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "pack.bin"
        target.write_bytes(b"payload")
        secret = tmp_path / "secret.key"
        secret.write_text("key", encoding="utf-8")

        class Result:
            returncode = 0

        with (
            patch("dlm.share.signing.minisign_available", return_value=True),
            patch("subprocess.run", return_value=Result()),
            pytest.raises(Exception, match="is missing"),
        ):
            sign_file(target, secret_key=secret)

    def test_happy_path_returns_minisig_path(self, tmp_path: Path) -> None:
        target = tmp_path / "pack.bin"
        target.write_bytes(b"payload")
        secret = tmp_path / "secret.key"
        secret.write_text("key", encoding="utf-8")
        sig = target.with_suffix(target.suffix + ".minisig")

        class Result:
            returncode = 0

        def _fake_run(cmd: list[str], check: bool) -> Result:
            assert "-c" in cmd
            sig.write_text("signature", encoding="utf-8")
            return Result()

        with (
            patch("dlm.share.signing.minisign_available", return_value=True),
            patch("subprocess.run", side_effect=_fake_run),
        ):
            out = sign_file(target, secret_key=secret, comment="demo")

        assert out == sig


class TestVerifySignature:
    def test_verified_when_one_key_matches(self, tmp_path: Path) -> None:
        target = tmp_path / "pack.bin"
        target.write_bytes(b"payload")
        sig = tmp_path / "pack.bin.minisig"
        sig.write_bytes(b"sig")
        keys = tmp_path / "trusted-keys"
        keys.mkdir()
        miss = keys / "miss.pub"
        hit = keys / "hit.pub"
        miss.write_text("miss", encoding="utf-8")
        hit.write_text("hit", encoding="utf-8")
        seen: list[Path] = []

        def _fake_verify(_target: Path, _sig: Path, pub_key: Path) -> None:
            seen.append(pub_key)
            if pub_key == miss:
                raise Exception("bad key")

        with (
            patch("dlm.share.signing.minisign_available", return_value=True),
            patch("dlm.share.signing._minisign_verify", side_effect=_fake_verify),
        ):
            result = verify_signature(target, sig, trusted_keys_dir=keys)

        assert result.status == VerifyStatus.VERIFIED
        assert result.key_path == hit
        assert seen == [hit]

    def test_unverified_when_no_keys_match(self, tmp_path: Path) -> None:
        target = tmp_path / "pack.bin"
        target.write_bytes(b"payload")
        sig = tmp_path / "pack.bin.minisig"
        sig.write_bytes(b"sig")
        keys = tmp_path / "trusted-keys"
        keys.mkdir()
        (keys / "a.pub").write_text("a", encoding="utf-8")
        (keys / "b.pub").write_text("b", encoding="utf-8")

        with (
            patch("dlm.share.signing.minisign_available", return_value=True),
            patch("dlm.share.signing._minisign_verify", side_effect=ShareError("no match")),
        ):
            result = verify_signature(target, sig, trusted_keys_dir=keys)

        assert result.status == VerifyStatus.UNVERIFIED
        assert "no match among 2 trusted keys" in result.detail


class TestMinisignVerify:
    def test_verify_raises_share_error_on_nonzero_exit(self, tmp_path: Path) -> None:
        from dlm.share.errors import ShareError
        from dlm.share.signing import _minisign_verify

        target = tmp_path / "pack.bin"
        target.write_bytes(b"payload")
        sig = tmp_path / "pack.bin.minisig"
        sig.write_bytes(b"sig")
        key = tmp_path / "key.pub"
        key.write_text("key", encoding="utf-8")

        class Result:
            returncode = 1
            stderr = b"bad signature"

        with (
            patch("subprocess.run", return_value=Result()),
            pytest.raises(ShareError, match="bad signature"),
        ):
            _minisign_verify(target, sig, key)
