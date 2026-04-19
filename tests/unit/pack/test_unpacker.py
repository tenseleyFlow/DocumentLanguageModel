"""`dlm.pack.unpacker` — extract + verify + header gate + install (Sprint 14)."""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

import pytest
import zstandard as zstd

from dlm.pack.errors import (
    PackFormatVersionError,
    PackIntegrityError,
    PackLayoutError,
)
from dlm.pack.integrity import rollup_sha256, write_checksums
from dlm.pack.layout import (
    HEADER_FILENAME,
    MANIFEST_FILENAME,
    SHA256_FILENAME,
)
from dlm.pack.unpacker import unpack


def _synth_pack(
    tmp_path: Path,
    *,
    pack_format_version: int = 1,
    content_override: bytes | None = None,
    skip_entry: str | None = None,
) -> Path:
    """Hand-assemble a tarball/zstd pack with full control over shape.

    Used to exercise the unpacker's header/layout/checksum gates
    without running the real packer (which always produces valid
    output).
    """
    staging = tmp_path / "stage"
    staging.mkdir()
    dlm_dir = staging / "dlm"
    dlm_dir.mkdir()
    (dlm_dir / "mydoc.dlm").write_text("---\ndlm_id: 01TEST\nbase_model: smollm2-135m\n---\n")

    store_dir = staging / "store"
    store_dir.mkdir()
    (store_dir / "manifest.json").write_text(
        json.dumps({"dlm_id": "01TEST", "base_model": "smollm2-135m"})
    )

    header = {
        "pack_format_version": pack_format_version,
        "created_at": "2026-04-19T12:00:00",
        "tool_version": "0.1.0",
        "content_type": "minimal",
        "platform_hint": "linux",
        "licensee_acceptance_url": None,
    }
    (staging / HEADER_FILENAME).write_text(json.dumps(header))

    checksums = write_checksums(staging, exclude=(SHA256_FILENAME, MANIFEST_FILENAME))
    pack_manifest = {
        "dlm_id": "01TEST",
        "base_model": "smollm2-135m",
        "base_model_revision": None,
        "base_model_sha256": None,
        "adapter_version": 0,
        "entries": {rel: (staging / rel).stat().st_size for rel in checksums},
        "content_sha256": rollup_sha256(checksums),
    }
    (staging / MANIFEST_FILENAME).write_text(json.dumps(pack_manifest))

    # Optional post-write mutations for negative tests.
    if content_override is not None:
        (staging / "dlm" / "mydoc.dlm").write_bytes(content_override)
    if skip_entry is not None:
        (staging / skip_entry).unlink()

    out = tmp_path / "synth.pack"
    cctx = zstd.ZstdCompressor(level=1)
    with out.open("wb") as fh, cctx.stream_writer(fh) as compressor:
        with tarfile.open(fileobj=compressor, mode="w|") as tar:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    tar.add(path, arcname=path.relative_to(staging).as_posix())
    return out


class TestHappyPath:
    def test_unpacks_into_dlm_home(self, tmp_path: Path) -> None:
        pack_path = _synth_pack(tmp_path)
        result = unpack(pack_path, home=tmp_path / "home", out_dir=tmp_path / "out")
        assert result.dlm_id == "01TEST"
        assert result.store_path == tmp_path / "home" / "store" / "01TEST"
        assert (result.store_path / "manifest.json").exists()
        assert result.dlm_path == tmp_path / "out" / "mydoc.dlm"
        assert result.dlm_path.read_text().startswith("---")


class TestVersionGate:
    def test_newer_than_current_refused(self, tmp_path: Path) -> None:
        pack_path = _synth_pack(tmp_path, pack_format_version=999)
        with pytest.raises(PackFormatVersionError):
            unpack(pack_path, home=tmp_path / "home")


class TestIntegrity:
    def test_corrupted_pack_refused(self, tmp_path: Path) -> None:
        """Build a pack whose .dlm content disagrees with the recorded checksum."""
        staging = tmp_path / "stage"
        staging.mkdir()
        (staging / "dlm").mkdir()
        (staging / "dlm" / "x.dlm").write_text("original")
        (staging / "store").mkdir()
        (staging / "store" / "manifest.json").write_text("{}")
        (staging / HEADER_FILENAME).write_text(
            json.dumps(
                {
                    "pack_format_version": 1,
                    "created_at": "2026-04-19T12:00:00",
                    "tool_version": "0.1.0",
                    "content_type": "minimal",
                    "platform_hint": "linux",
                    "licensee_acceptance_url": None,
                }
            )
        )

        # Freeze checksums against the original content, then tamper.
        write_checksums(staging, exclude=(SHA256_FILENAME, MANIFEST_FILENAME))
        (staging / "dlm" / "x.dlm").write_text("TAMPERED!")

        # Manifest is required for layout but its content_sha256 isn't
        # rechecked at unpack (integrity gate catches the tamper first).
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "dlm_id": "01TEST",
                    "base_model": "smollm2-135m",
                    "base_model_revision": None,
                    "base_model_sha256": None,
                    "adapter_version": 0,
                    "entries": {},
                    "content_sha256": "0" * 64,
                }
            )
        )

        out = tmp_path / "tampered.pack"
        cctx = zstd.ZstdCompressor(level=1)
        with out.open("wb") as fh, cctx.stream_writer(fh) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                for path in sorted(staging.rglob("*")):
                    if path.is_file():
                        tar.add(path, arcname=path.relative_to(staging).as_posix())

        with pytest.raises(PackIntegrityError):
            unpack(out, home=tmp_path / "home")


class TestLayoutGate:
    def test_missing_header_refused(self, tmp_path: Path) -> None:
        # Build a tarball without PACK_HEADER.json.
        staging = tmp_path / "stage"
        staging.mkdir()
        (staging / "dlm").mkdir()
        (staging / "dlm" / "x.dlm").write_text("x")
        (staging / "store").mkdir()
        (staging / "store" / "manifest.json").write_text("{}")
        (staging / MANIFEST_FILENAME).write_text("{}")
        write_checksums(staging, exclude=(SHA256_FILENAME, MANIFEST_FILENAME))
        out = tmp_path / "no-header.pack"
        cctx = zstd.ZstdCompressor(level=1)
        with out.open("wb") as fh, cctx.stream_writer(fh) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                for path in sorted(staging.rglob("*")):
                    if path.is_file():
                        tar.add(path, arcname=path.relative_to(staging).as_posix())
        with pytest.raises(PackLayoutError):
            unpack(out, home=tmp_path / "home")

    def test_oversized_member_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Audit-04 B7: a tar entry exceeding the per-member cap is rejected
        before we touch disk. Monkeypatches the cap down to a tiny value
        so we can build a real fixture without writing 16 GiB."""
        import dlm.pack.unpacker as unpacker_mod

        monkeypatch.setattr(unpacker_mod, "_MAX_TAR_MEMBER_BYTES", 10)

        staging = tmp_path / "s"
        staging.mkdir()
        # 100 bytes — 10× the patched cap.
        big = staging / "big"
        big.write_bytes(b"A" * 100)

        out = tmp_path / "bomb.pack"
        cctx = zstd.ZstdCompressor(level=1)
        with out.open("wb") as fh, cctx.stream_writer(fh) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                tar.add(big, arcname="big")

        with pytest.raises(PackLayoutError, match="exceeds per-member cap"):
            unpack(out, home=tmp_path / "home")

    def test_total_decompressed_size_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Audit-04 B7: the cumulative size cap fires even when each member
        fits under the per-member cap."""
        import dlm.pack.unpacker as unpacker_mod

        monkeypatch.setattr(unpacker_mod, "_MAX_TAR_MEMBER_BYTES", 100)
        monkeypatch.setattr(unpacker_mod, "_MAX_PACK_DECOMPRESSED_BYTES", 150)

        staging = tmp_path / "s"
        staging.mkdir()
        # Two 100-byte members — each under the cap, together over it.
        (staging / "a").write_bytes(b"A" * 100)
        (staging / "b").write_bytes(b"B" * 100)

        out = tmp_path / "two.pack"
        cctx = zstd.ZstdCompressor(level=1)
        with out.open("wb") as fh, cctx.stream_writer(fh) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                for name in ("a", "b"):
                    tar.add(staging / name, arcname=name)

        with pytest.raises(PackLayoutError, match="total decompressed size"):
            unpack(out, home=tmp_path / "home")

    def test_symlink_member_blocked_by_data_filter(self, tmp_path: Path) -> None:
        """Python's `data` extraction filter refuses symlink members entirely
        (audit-04 T5). We double-check the layout scan accepts them (they're
        not path-traversing) but extraction itself blocks the write."""
        staging = tmp_path / "s"
        staging.mkdir()
        (staging / "real.txt").write_bytes(b"real")

        out = tmp_path / "symlink.pack"
        cctx = zstd.ZstdCompressor(level=1)
        with out.open("wb") as fh, cctx.stream_writer(fh) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                info = tarfile.TarInfo(name="link_to_real")
                info.type = tarfile.SYMTYPE
                info.linkname = "real.txt"
                tar.addfile(info)

        # The data filter raises on symlinks; pipeline surfaces the
        # failure up as a tarfile error that the caller sees.
        with pytest.raises(Exception):  # noqa: B017, PT011
            unpack(out, home=tmp_path / "home")

    def test_duplicate_entry_name_refused(self, tmp_path: Path) -> None:
        """Two tar members with the same name must be refused up-front.

        Audit-05 M5: the prior version of this test planted *differing*
        content, which tripped the CHECKSUMS verify by accident. An
        attacker who repacks with identical-content duplicates could
        slip past that. The first-pass scan now refuses duplicates
        regardless of content — verified by the second test below.
        """
        # Build a pack that would otherwise be valid, then repack with a
        # duplicated entry, re-extract, and check pipeline rejects.
        staging = tmp_path / "s"
        staging.mkdir()
        (staging / "dlm").mkdir()
        (staging / "dlm" / "a.dlm").write_text("x")
        (staging / "store").mkdir()
        (staging / "store" / "manifest.json").write_text("{}")
        (staging / HEADER_FILENAME).write_text(
            json.dumps(
                {
                    "pack_format_version": 1,
                    "created_at": "2026-04-19T12:00:00",
                    "tool_version": "0.1.0",
                    "content_type": "minimal",
                    "platform_hint": "linux",
                    "licensee_acceptance_url": None,
                }
            )
        )
        write_checksums(staging, exclude=(SHA256_FILENAME, MANIFEST_FILENAME))
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "dlm_id": "01TEST",
                    "base_model": "smollm2-135m",
                    "base_model_revision": None,
                    "base_model_sha256": None,
                    "adapter_version": 0,
                    "entries": {},
                    "content_sha256": "0" * 64,
                }
            )
        )

        out = tmp_path / "dup.pack"
        cctx = zstd.ZstdCompressor(level=1)
        with out.open("wb") as fh, cctx.stream_writer(fh) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                for path in sorted(staging.rglob("*")):
                    if path.is_file():
                        tar.add(path, arcname=path.relative_to(staging).as_posix())
                # Duplicate `dlm/a.dlm` with different content.
                info = tarfile.TarInfo(name="dlm/a.dlm")
                info.size = 1
                import io as _io

                tar.addfile(info, _io.BytesIO(b"Z"))

        # Extraction overwrites the checksummed file; the first-pass
        # layout scan refuses the duplicate before we even touch disk.
        with pytest.raises(PackLayoutError, match="duplicate tar entry"):
            unpack(out, home=tmp_path / "home")

    def test_duplicate_entry_same_content_also_refused(self, tmp_path: Path) -> None:
        """Audit-05 M5: even duplicates with identical content are refused.

        The CHECKSUMS-based defense in the original audit-04 test would
        silently accept this case because the final on-disk content
        hashes match the recorded sums. Defense-in-depth lives in the
        first-pass duplicate-name scan.
        """
        staging = tmp_path / "s"
        staging.mkdir()
        (staging / "dlm").mkdir()
        (staging / "dlm" / "a.dlm").write_text("x")
        (staging / "store").mkdir()
        (staging / "store" / "manifest.json").write_text("{}")
        (staging / HEADER_FILENAME).write_text(
            json.dumps(
                {
                    "pack_format_version": 1,
                    "created_at": "2026-04-19T12:00:00",
                    "tool_version": "0.1.0",
                    "content_type": "minimal",
                    "platform_hint": "linux",
                    "licensee_acceptance_url": None,
                }
            )
        )
        write_checksums(staging, exclude=(SHA256_FILENAME, MANIFEST_FILENAME))
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "dlm_id": "01TEST",
                    "base_model": "smollm2-135m",
                    "base_model_revision": None,
                    "base_model_sha256": None,
                    "adapter_version": 0,
                    "entries": {},
                    "content_sha256": "0" * 64,
                }
            )
        )

        out = tmp_path / "dup-identical.pack"
        cctx = zstd.ZstdCompressor(level=1)
        with out.open("wb") as fh, cctx.stream_writer(fh) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                for path in sorted(staging.rglob("*")):
                    if path.is_file():
                        tar.add(path, arcname=path.relative_to(staging).as_posix())
                # Duplicate `dlm/a.dlm` with byte-identical content.
                info = tarfile.TarInfo(name="dlm/a.dlm")
                info.size = 1
                import io as _io

                tar.addfile(info, _io.BytesIO(b"x"))

        with pytest.raises(PackLayoutError, match="duplicate tar entry"):
            unpack(out, home=tmp_path / "home")

    def test_unsafe_tar_entry_refused(self, tmp_path: Path) -> None:
        """An entry whose path escapes the extraction root is rejected."""
        staging = tmp_path / "s"
        staging.mkdir()
        evil = staging / "evil"
        evil.write_text("x")
        out = tmp_path / "evil.pack"
        cctx = zstd.ZstdCompressor(level=1)
        with out.open("wb") as fh, cctx.stream_writer(fh) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                info = tar.gettarinfo(evil, arcname="../escape")
                with evil.open("rb") as src:
                    tar.addfile(info, src)
        with pytest.raises(PackLayoutError):
            unpack(out, home=tmp_path / "home")


class TestForce:
    def test_existing_store_refused_without_force(self, tmp_path: Path) -> None:
        pack_path = _synth_pack(tmp_path)
        home = tmp_path / "home"
        unpack(pack_path, home=home, out_dir=tmp_path / "out1")
        # Second unpack without force fails because the store exists.
        with pytest.raises(PackIntegrityError):
            unpack(pack_path, home=home, out_dir=tmp_path / "out2")

    def test_force_replaces_existing_store(self, tmp_path: Path) -> None:
        pack_path = _synth_pack(tmp_path)
        home = tmp_path / "home"
        unpack(pack_path, home=home, out_dir=tmp_path / "out1")
        # Add a marker to the prior store; --force should wipe it.
        marker = home / "store" / "01TEST" / "marker.txt"
        marker.write_text("prior")
        unpack(pack_path, home=home, force=True, out_dir=tmp_path / "out2")
        assert not marker.exists()

    def test_move_failure_rolls_back_to_prior_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Audit-04 B6: interrupted install must not leave caller with no store.

        Before the fix: `rmtree(old); move(new)` — failure between them
        wipes the old store and leaves a gap. After: old is quarantined,
        move is attempted, and on failure the quarantine is restored.
        """
        import shutil as shutil_mod

        pack_path = _synth_pack(tmp_path)
        home = tmp_path / "home"
        unpack(pack_path, home=home, out_dir=tmp_path / "out1")
        target = home / "store" / "01TEST"
        (target / "sentinel.txt").write_text("pre-existing")

        # Force the move to fail — simulating power loss / OOM mid-install.
        def _boom(*_a: object, **_kw: object) -> None:
            raise OSError("simulated crash during move")

        monkeypatch.setattr(shutil_mod, "move", _boom)

        with pytest.raises(OSError, match="simulated crash"):
            unpack(pack_path, home=home, force=True, out_dir=tmp_path / "out2")

        # Prior store must still be present and intact.
        assert target.exists()
        assert (target / "sentinel.txt").read_text() == "pre-existing"
        # No orphan quarantine left behind.
        orphans = [p for p in target.parent.iterdir() if p.name.startswith(".01TEST.old-")]
        assert orphans == [], f"quarantine not cleaned up: {orphans}"


# Keep tempfile import reachable for downstream contributors extending this file.
_ = tempfile
