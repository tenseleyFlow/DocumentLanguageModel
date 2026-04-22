"""`dlm.pack.packer` — pack creation + content-type labelling + F21 gate (Sprint 14)."""

from __future__ import annotations

import logging
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app
from dlm.pack.errors import BaseLicenseRefusedError, PackExecutableFileError
from dlm.pack.packer import _platform_hint, pack


def _scaffold_doc_and_store(tmp_path: Path, *, base: str = "smollm2-135m") -> Path:
    """Create a .dlm + its ensure_layout'd store with a valid manifest."""
    from dlm.doc.parser import parse_file
    from dlm.store.manifest import Manifest, save_manifest
    from dlm.store.paths import for_dlm

    runner = CliRunner()
    doc = tmp_path / "doc.dlm"
    # Gated bases (e.g. llama-3.2) require --i-accept-license in non-interactive runs.
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_path / "dlm-home"),
            "init",
            str(doc),
            "--base",
            base,
            "--i-accept-license",
        ],
    )
    assert result.exit_code == 0, result.output

    parsed = parse_file(doc)
    import os

    os.environ["DLM_HOME"] = str(tmp_path / "dlm-home")
    store = for_dlm(parsed.frontmatter.dlm_id)
    store.ensure_layout()
    save_manifest(store.manifest, Manifest(dlm_id=parsed.frontmatter.dlm_id, base_model=base))
    return doc


def _tar_members(pack_path: Path) -> list[tuple[str, int]]:
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    with pack_path.open("rb") as fh, dctx.stream_reader(fh) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            return [(m.name, m.size) for m in tar]


class TestPackShape:
    def test_contains_required_entries(self, tmp_path: Path) -> None:
        doc = _scaffold_doc_and_store(tmp_path)
        result = pack(doc)

        names = {name for name, _size in _tar_members(result.path)}
        assert "PACK_HEADER.json" in names
        assert "manifest.json" in names
        assert "CHECKSUMS.sha256" in names
        assert f"dlm/{doc.name}" in names
        assert "store/manifest.json" in names

    def test_default_out_path_next_to_dlm(self, tmp_path: Path) -> None:
        doc = _scaffold_doc_and_store(tmp_path)
        result = pack(doc)
        assert result.path == doc.with_suffix(doc.suffix + ".pack")
        assert result.path.is_file()

    def test_explicit_out_path(self, tmp_path: Path) -> None:
        doc = _scaffold_doc_and_store(tmp_path)
        custom = tmp_path / "sub" / "custom.pack"
        result = pack(doc, out=custom)
        assert result.path == custom
        assert custom.is_file()


class TestContentTypeLabel:
    def test_minimal_default(self, tmp_path: Path) -> None:
        doc = _scaffold_doc_and_store(tmp_path)
        result = pack(doc)
        assert result.content_type == "minimal"

    def test_include_exports_flips_to_no_base(self, tmp_path: Path) -> None:
        doc = _scaffold_doc_and_store(tmp_path)
        result = pack(doc, include_exports=True)
        assert result.content_type == "no-base"

    def test_include_base_flips_to_no_exports(self, tmp_path: Path) -> None:
        doc = _scaffold_doc_and_store(tmp_path)
        result = pack(doc, include_base=True)
        assert result.content_type == "no-exports"

    def test_both_flags_equal_full(self, tmp_path: Path) -> None:
        doc = _scaffold_doc_and_store(tmp_path)
        result = pack(doc, include_base=True, include_exports=True)
        assert result.content_type == "full"


class TestRedistributionGate:
    """Audit F21: --include-base on non-redistributable spec refuses unless licensee URL."""

    def test_refuses_without_licensee_url(self, tmp_path: Path) -> None:
        doc = _scaffold_doc_and_store(tmp_path, base="llama-3.2-1b")
        with pytest.raises(BaseLicenseRefusedError) as excinfo:
            pack(doc, include_base=True)
        assert excinfo.value.base_key == "llama-3.2-1b"

    def test_accepts_with_licensee_url(self, tmp_path: Path) -> None:
        doc = _scaffold_doc_and_store(tmp_path, base="llama-3.2-1b")
        result = pack(
            doc,
            include_base=True,
            licensee_acceptance_url="https://example.com/accept",
        )
        assert result.applied_licensee_url == "https://example.com/accept"
        # Header records the URL.
        import json

        import zstandard as zstd

        dctx = zstd.ZstdDecompressor()
        with result.path.open("rb") as fh, dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    if member.name == "PACK_HEADER.json":
                        extracted = tar.extractfile(member)
                        assert extracted is not None
                        header = json.loads(extracted.read())
                        assert header["licensee_acceptance_url"] == "https://example.com/accept"
                        return
        pytest.fail("PACK_HEADER.json not found in tar")

    def test_redistributable_base_needs_no_licensee(self, tmp_path: Path) -> None:
        """Apache-licensed bases like smollm2 don't gate on the URL."""
        doc = _scaffold_doc_and_store(tmp_path, base="smollm2-135m")
        result = pack(doc, include_base=True)  # no licensee url
        assert result.content_type == "no-exports"


class TestByteIdentical:
    """Audit-04 B5: two packs of identical input bytes produce identical pack bytes.

    Pins `datetime.now` (PackHeader timestamp) and relies on the
    normalized TarInfo filter (no mtime / uid / gid / uname / gname)
    added by this fix.
    """

    def test_same_content_produces_same_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime

        import dlm.pack.packer as packer_mod

        frozen = datetime(2026, 4, 19, 12, 0, 0, tzinfo=UTC)

        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
                return frozen

        monkeypatch.setattr(packer_mod, "datetime", _FrozenDatetime)

        doc = _scaffold_doc_and_store(tmp_path)
        a = pack(doc, out=tmp_path / "a.pack")
        b = pack(doc, out=tmp_path / "b.pack")

        assert a.path.read_bytes() == b.path.read_bytes()


class TestStoreLock:
    def test_acquires_lock_during_pack(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The store exclusive lock must be held for the duration of the copy.

        Audit-04 T1: prior test just checked `lock.exists()`, which is
        satisfied by any stale lockfile. This variant asserts the *live*
        exclusive semantics — a concurrent `exclusive(store.lock)`
        attempt during the copy must block (raise LockHeldError under
        a small timeout)."""
        doc = _scaffold_doc_and_store(tmp_path)

        from dlm.doc.parser import parse_file
        from dlm.store.errors import LockHeldError
        from dlm.store.lock import exclusive
        from dlm.store.paths import for_dlm

        parsed = parse_file(doc)
        store = for_dlm(parsed.frontmatter.dlm_id)

        observed: dict[str, object] = {}

        original_copytree = __import__("shutil").copytree

        def watcher(*args: Any, **kwargs: Any) -> Any:
            # Mid-copy: try to acquire the same lock. It must already be
            # held by the packer, so `exclusive(..., timeout=0)` raises.
            try:
                with exclusive(store.lock, timeout=0.0):
                    observed["second_acquire"] = "succeeded"
            except LockHeldError as exc:
                observed["second_acquire"] = "blocked"
                observed["holder_pid"] = exc.holder_pid
            return original_copytree(*args, **kwargs)

        monkeypatch.setattr("shutil.copytree", watcher)
        pack(doc)

        import os as _os

        assert observed.get("second_acquire") == "blocked"
        # The packer is the same process — holder_pid matches ours.
        assert observed.get("holder_pid") == _os.getpid()


class TestExecutableBitRefusal:
    def test_refuses_executable_file_in_store_tree(self, tmp_path: Path) -> None:
        doc = _scaffold_doc_and_store(tmp_path)

        from dlm.doc.parser import parse_file
        from dlm.store.paths import for_dlm

        parsed = parse_file(doc)
        store = for_dlm(parsed.frontmatter.dlm_id)
        hook = store.root / "resume.sh"
        hook.write_text("#!/bin/sh\necho nope\n", encoding="utf-8")
        hook.chmod(0o755)

        with pytest.raises(PackExecutableFileError, match="resume\\.sh"):
            pack(doc)


class TestPlatformHint:
    def test_import_error_falls_back_to_sys_platform(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)

        def _boom() -> object:
            raise ImportError("torch unavailable")

        assert _platform_hint(detect_backend=_boom) == sys.platform
        assert "falling back to" in caplog.text

    def test_oserror_falls_back_to_unknown(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)

        def _boom() -> object:
            raise OSError("cuda init failed")

        assert _platform_hint(detect_backend=_boom) == f"{sys.platform}-unknown"
        assert "unknown backend" in caplog.text
