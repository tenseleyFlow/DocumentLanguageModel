"""`dlm verify` CLI — exit codes + messaging for each failure mode."""

from __future__ import annotations

import io
import re
import tarfile
from pathlib import Path

import zstandard as zstd
from typer.testing import CliRunner

from dlm.cli.app import app
from dlm.pack.layout import PROVENANCE_FILENAME

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _normalized_output(text: str) -> str:
    plain = _ANSI_RE.sub("", text)
    tableless = plain.translate(str.maketrans(dict.fromkeys("│╭╮╰╯─", " ")))
    return " ".join(tableless.split())


def _make_pack_with(payloads: dict[str, bytes], path: Path) -> None:
    """Write a minimal tar+zstd pack containing `payloads` by name.

    The pack isn't a valid full `.dlm.pack` (missing headers/checksums),
    but `read_pack_member_bytes` only iterates members — it doesn't
    re-check layout — so this is enough for the verify CLI's provenance
    lookup. Full-pack tests live elsewhere.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in payloads.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    cctx = zstd.ZstdCompressor()
    path.write_bytes(cctx.compress(raw))


def test_verify_missing_provenance_exits_1(tmp_path: Path) -> None:
    pack = tmp_path / "unsigned.dlm.pack"
    _make_pack_with({"OTHER_FILE.txt": b"nope"}, pack)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["verify", str(pack), "--trusted-keys-dir", str(tmp_path / "keys")],
    )
    assert result.exit_code == 1
    assert "unsigned" in result.output.lower() or "unsigned" in result.stderr.lower()


def test_verify_malformed_provenance_exits_1(tmp_path: Path) -> None:
    pack = tmp_path / "bad.dlm.pack"
    _make_pack_with({PROVENANCE_FILENAME: b"{not json"}, pack)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["verify", str(pack), "--trusted-keys-dir", str(tmp_path / "keys")],
    )
    assert result.exit_code == 1


def test_verify_unknown_signer_strict_exits_2(tmp_path: Path) -> None:
    """Strict mode (default) refuses unknown signer with exit 2."""
    import json

    provenance = {
        "adapter_sha256": "a" * 64,
        "base_revision": "b" * 40,
        "corpus_root_sha256": "c" * 64,
        "env_lock_digest": "d" * 64,
        "signed_at": "2026-04-21T12:00:00Z",
        "signer_public_key": "untrusted comment: minisign public key\nRWSUNKNOWNSIGNERPUBKEY123==\n",
        "signature": "untrusted comment: signature\nZZZZ==\ntrusted comment: x",
    }
    pack = tmp_path / "unknown-signer.dlm.pack"
    _make_pack_with({PROVENANCE_FILENAME: json.dumps(provenance).encode("utf-8")}, pack)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["verify", str(pack), "--trusted-keys-dir", str(tmp_path / "keys")],
    )
    assert result.exit_code == 2
    assert "signer" in result.output.lower() or "signer" in result.stderr.lower()


def test_verify_help_text_surfaces_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["verify", "--help"])
    assert result.exit_code == 0
    text = _normalized_output(result.output)
    assert "--trust-on-first-use" in text
    assert "provenance" in text.lower()
