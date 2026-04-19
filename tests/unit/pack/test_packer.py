"""`dlm.pack.packer` — pack creation + content-type labelling + F21 gate (Sprint 14)."""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app
from dlm.pack.errors import BaseLicenseRefusedError
from dlm.pack.packer import pack


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


class TestStoreLock:
    def test_acquires_lock_during_pack(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The store exclusive lock must be held for the duration of the copy."""
        doc = _scaffold_doc_and_store(tmp_path)

        from dlm.doc.parser import parse_file
        from dlm.store.paths import for_dlm

        parsed = parse_file(doc)
        store = for_dlm(parsed.frontmatter.dlm_id)

        seen_lock_state: dict[str, bool] = {}

        original_copytree = __import__("shutil").copytree

        def watcher(*args: Any, **kwargs: Any) -> Any:
            seen_lock_state["locked"] = store.lock.exists()
            return original_copytree(*args, **kwargs)

        monkeypatch.setattr("shutil.copytree", watcher)
        pack(doc)
        assert seen_lock_state.get("locked") is True
