"""End-to-end round-trip: init → pack → delete store → unpack → verify (Sprint 14).

Sprint 14 DoD §1: "pack a tiny-model store → delete original → unpack →
`dlm prompt` works and produces the same output as before pack". This
integration omits the `dlm prompt` step (requires a real HF model load
which is slow-marker-gated) and focuses on the structural guarantees:

- Pack preserves the `.dlm` byte-identical.
- Unpack restores `store/manifest.json` byte-identical.
- dlm_id matches across the round trip.
- The pack's own artifact is reproducible: same inputs → same
  `content_sha256` in the recorded pack manifest.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import pytest
import zstandard as zstd
from typer.testing import CliRunner

from dlm.cli.app import app


def _scaffolded_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Init a doc, ensure_layout the store, save a manifest — return the .dlm path.

    Audit-05 N11: `monkeypatch.setenv` instead of raw `os.environ[...]`
    so the DLM_HOME override auto-reverts at test teardown and can't
    leak into a later test in the same session.
    """
    from dlm.doc.parser import parse_file
    from dlm.store.paths import for_dlm

    home = tmp_path / "dlm-home"
    monkeypatch.setenv("DLM_HOME", str(home))
    runner = CliRunner()
    doc = tmp_path / "doc.dlm"
    result = runner.invoke(
        app,
        [
            "--home",
            str(home),
            "init",
            str(doc),
            "--base",
            "smollm2-135m",
        ],
    )
    assert result.exit_code == 0, result.output

    # Post-B2: `dlm init` already creates the store + manifest. We just
    # drop a marker file so round-trip tests cover non-manifest content.
    parsed = parse_file(doc)
    store = for_dlm(parsed.frontmatter.dlm_id)
    (store.root / "marker.txt").write_text("i survived the round trip\n")
    return doc


class TestRoundTrip:
    def test_pack_unpack_preserves_dlm_and_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dlm.pack.packer import pack
        from dlm.pack.unpacker import unpack

        doc = _scaffolded_store(tmp_path, monkeypatch)
        original_dlm = doc.read_bytes()

        # Capture pre-pack store state.
        from dlm.doc.parser import parse_file
        from dlm.store.paths import for_dlm

        parsed = parse_file(doc)
        store = for_dlm(parsed.frontmatter.dlm_id)
        original_manifest = store.manifest.read_bytes()
        original_marker = (store.root / "marker.txt").read_bytes()
        dlm_id = parsed.frontmatter.dlm_id

        # Pack.
        pack_result = pack(doc, out=tmp_path / "out.pack")
        assert pack_result.path.is_file()
        assert pack_result.content_type == "minimal"

        # Destroy the original store + doc to simulate a fresh machine.
        shutil.rmtree(store.root)
        doc.unlink()

        # Unpack into a fresh home so there's no chance of crossover.
        fresh_home = tmp_path / "fresh-home"
        monkeypatch.setenv("DLM_HOME", str(fresh_home))
        unpack_result = unpack(pack_result.path, home=fresh_home, out_dir=tmp_path / "restored")

        # Assertions — structural invariants.
        assert unpack_result.dlm_id == dlm_id
        assert unpack_result.dlm_path.read_bytes() == original_dlm
        restored_manifest = (unpack_result.store_path / "manifest.json").read_bytes()
        assert restored_manifest == original_manifest
        restored_marker = (unpack_result.store_path / "marker.txt").read_bytes()
        assert restored_marker == original_marker

    def test_pack_manifest_content_sha256_is_deterministic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two packs of identical input produce identical `content_sha256` rollups."""
        from dlm.pack.packer import pack

        doc = _scaffolded_store(tmp_path, monkeypatch)
        a = pack(doc, out=tmp_path / "a.pack")
        b = pack(doc, out=tmp_path / "b.pack")

        def read_manifest(pack_path: Path) -> dict[str, object]:
            dctx = zstd.ZstdDecompressor()
            with pack_path.open("rb") as fh, dctx.stream_reader(fh) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    for m in tar:
                        if m.name == "manifest.json":
                            f = tar.extractfile(m)
                            assert f is not None
                            data: dict[str, object] = json.loads(f.read())
                            return data
            pytest.fail("manifest.json missing")

        a_rollup = read_manifest(a.path)["content_sha256"]
        b_rollup = read_manifest(b.path)["content_sha256"]
        assert a_rollup == b_rollup


class TestForceOverwrite:
    def test_unpack_force_replaces_existing_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dlm.pack.packer import pack
        from dlm.pack.unpacker import unpack

        doc = _scaffolded_store(tmp_path, monkeypatch)
        pack_result = pack(doc, out=tmp_path / "out.pack")

        fresh_home = tmp_path / "fresh-home"
        monkeypatch.setenv("DLM_HOME", str(fresh_home))
        first = unpack(pack_result.path, home=fresh_home, out_dir=tmp_path / "out1")

        # Add a marker AFTER unpack; --force should wipe it.
        marker = first.store_path / "post-unpack-marker.txt"
        marker.write_text("should not survive --force")

        second = unpack(
            pack_result.path,
            home=fresh_home,
            force=True,
            out_dir=tmp_path / "out2",
        )
        assert second.store_path == first.store_path
        assert not marker.exists()
