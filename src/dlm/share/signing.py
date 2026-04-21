"""Optional pack signing via the `minisign` CLI tool.

`minisign` (https://jedisct1.github.io/minisign/) is a small,
well-audited signing tool that uses Ed25519. We wrap it via subprocess
rather than reimplementing crypto — matches the sprint spec exactly
("Optional. `dlm push --sign` signs the pack's CHECKSUMS.sha256 with
minisign") and avoids adding a heavy crypto dep to dlm's runtime.

Flow:

1. **Sign** (push-side, `--sign`):
   - Locate `minisign` on PATH; refuse cleanly if missing.
   - Sign the pack's embedded `CHECKSUMS.sha256` with the user's
     minisign secret key (default `~/.dlm/minisign.key`, prompt for
     passphrase via minisign's own TTY path).
   - The resulting `.sig` file travels inside the pack next to the
     checksums.

2. **Verify** (pull-side):
   - If a signature is present, try every public key in
     `~/.dlm/trusted-keys/*.pub` until one verifies.
   - On match → `VerifyResult(status=VERIFIED, key_path=...)`.
   - On all-fail → `VerifyResult(status=UNVERIFIED)` + warn.
   - No signature → `VerifyResult(status=UNSIGNED)`, no warning.

Unsigned packs remain fully functional; signatures are additive trust,
not a gate.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from dlm.share.errors import ShareError

_LOG = logging.getLogger(__name__)

_DEFAULT_SECRET_KEY = Path.home() / ".dlm" / "minisign.key"
_DEFAULT_TRUSTED_KEYS_DIR = Path.home() / ".dlm" / "trusted-keys"


class MinisignNotAvailableError(ShareError):
    """`minisign` CLI isn't installed on PATH."""


class VerifyStatus(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    UNSIGNED = "unsigned"


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of `verify_pack_signature`.

    `key_path` is populated only when status is VERIFIED — it's the
    trusted public key file that matched. Callers print this in the
    "pulled + verified by ..." message.
    """

    status: VerifyStatus
    key_path: Path | None = None
    detail: str = ""


def minisign_available() -> bool:
    """Probe: is the `minisign` CLI on PATH?"""
    return shutil.which("minisign") is not None


def sign_file(
    target: Path,
    *,
    secret_key: Path | None = None,
    comment: str | None = None,
) -> Path:
    """Sign `target` with minisign; returns the `.minisig` file path.

    Uses `minisign -Sm <target>` which writes `<target>.minisig`
    alongside. The user's passphrase is prompted via minisign's own
    TTY path — we don't capture or store it.

    Raises `MinisignNotAvailableError` if the binary is missing.
    """
    if not minisign_available():
        raise MinisignNotAvailableError(
            "`minisign` is not installed. Install it with "
            "`brew install minisign` (macOS) or your system package "
            "manager, then retry with --sign."
        )

    secret = secret_key or _DEFAULT_SECRET_KEY
    if not secret.is_file():
        raise ShareError(
            f"minisign secret key not found at {secret}. Generate one with "
            f"`minisign -G -s {secret}` (prompts for passphrase)."
        )

    cmd = ["minisign", "-Sm", str(target), "-s", str(secret)]
    if comment is not None:
        cmd += ["-c", comment]

    _LOG.info("signing: minisign -Sm %s", target)
    # `minisign` prompts for passphrase on /dev/tty — don't capture
    # stdin/stdout so the user sees the prompt.
    result = subprocess.run(cmd, check=False)  # noqa: S603 — known binary
    if result.returncode != 0:
        raise ShareError(
            f"minisign refused to sign (exit {result.returncode}). "
            "Check your passphrase + key file."
        )

    sig_path = target.with_suffix(target.suffix + ".minisig")
    if not sig_path.is_file():
        raise ShareError(
            f"minisign succeeded but {sig_path} is missing; "
            "signing integration is out of sync"
        )
    return sig_path


def verify_signature(
    target: Path,
    signature: Path,
    *,
    trusted_keys_dir: Path | None = None,
) -> VerifyResult:
    """Verify `signature` against `target` using trusted public keys.

    Walks `trusted_keys_dir` (default `~/.dlm/trusted-keys/`) for
    `*.pub` files, trying each until one verifies. No directory →
    `UNVERIFIED` with a pointer to the setup docs.
    """
    if not signature.is_file():
        return VerifyResult(status=VerifyStatus.UNSIGNED)

    if not minisign_available():
        _LOG.warning(
            "verify: signature present (%s) but `minisign` is not installed; "
            "cannot verify",
            signature,
        )
        return VerifyResult(
            status=VerifyStatus.UNVERIFIED,
            detail="minisign not installed",
        )

    keys_dir = trusted_keys_dir or _DEFAULT_TRUSTED_KEYS_DIR
    if not keys_dir.is_dir():
        _LOG.warning(
            "verify: signature present but no trusted keys at %s; "
            "pack is unverifiable",
            keys_dir,
        )
        return VerifyResult(
            status=VerifyStatus.UNVERIFIED,
            detail=f"no trusted keys at {keys_dir}",
        )

    candidates = sorted(keys_dir.glob("*.pub"))
    for pub_key in candidates:
        try:
            _minisign_verify(target, signature, pub_key)
        except ShareError:
            continue
        return VerifyResult(status=VerifyStatus.VERIFIED, key_path=pub_key)

    _LOG.warning(
        "verify: signature present but no trusted key in %s matched (%d tried)",
        keys_dir,
        len(candidates),
    )
    return VerifyResult(
        status=VerifyStatus.UNVERIFIED,
        detail=f"no match among {len(candidates)} trusted keys",
    )


def _minisign_verify(target: Path, signature: Path, public_key: Path) -> None:
    """Run `minisign -Vm <target> -x <sig> -p <key>`; raise on failure."""
    cmd = [
        "minisign",
        "-Vm",
        str(target),
        "-x",
        str(signature),
        "-p",
        str(public_key),
    ]
    result = subprocess.run(  # noqa: S603 — known binary
        cmd,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ShareError(
            f"minisign -Vm: exit {result.returncode} "
            f"(stderr: {result.stderr.decode(errors='replace').strip()})"
        )
