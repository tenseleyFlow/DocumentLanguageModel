"""imatrix build + cache + argv (Sprint 11.6)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from dlm.export.errors import SubprocessError
from dlm.export.imatrix import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNKS,
    ImatrixArtifact,
    _sha256_of_file,
    build_imatrix,
    build_imatrix_args,
    calibration_text_from_replay,
    resolve_imatrix,
)


def _write_vendor(tmp_path: Path) -> Path:
    """Build a fake vendor tree so llama_imatrix_bin() resolves."""
    vendor = tmp_path / "vendor" / "llama.cpp"
    (vendor / "build" / "bin").mkdir(parents=True)
    (vendor / "build" / "bin" / "llama-imatrix").write_text("# mock")
    (vendor / "convert_hf_to_gguf.py").write_text("# mock")  # required by vendoring
    return vendor


# --- build_imatrix_args -------------------------------------------------------


class TestBuildArgs:
    def test_includes_all_required_flags(self, tmp_path: Path) -> None:
        vendor = _write_vendor(tmp_path)
        argv = build_imatrix_args(
            base_gguf=tmp_path / "base.fp16.gguf",
            calib_path=tmp_path / "calib.txt",
            out_path=tmp_path / "imatrix.gguf",
            chunks=128,
            bin_override=vendor,
        )
        # binary absolute path
        assert argv[0].endswith("/bin/llama-imatrix") or argv[0].endswith("llama-imatrix")
        assert "-m" in argv
        assert "-f" in argv
        assert "-o" in argv
        assert "--chunks" in argv
        assert "128" in argv

    def test_paths_are_stringified(self, tmp_path: Path) -> None:
        vendor = _write_vendor(tmp_path)
        argv = build_imatrix_args(
            base_gguf=tmp_path / "b.gguf",
            calib_path=tmp_path / "c.txt",
            out_path=tmp_path / "o.gguf",
            chunks=64,
            bin_override=vendor,
        )
        assert all(isinstance(a, str) for a in argv)


# --- build_imatrix ------------------------------------------------------------


class TestBuildImatrix:
    def _fake_runner(self, *, write_output: bool = True) -> tuple[Any, list[list[str]]]:
        """Runner that captures argv and optionally writes the output file."""
        calls: list[list[str]] = []

        def runner(argv: Any) -> None:
            cmd = list(argv)
            calls.append(cmd)
            if write_output:
                out_ix = cmd.index("-o") + 1
                Path(cmd[out_ix]).write_bytes(b"fake imatrix bytes")

        return runner, calls

    def test_successful_build_writes_sidecar(self, tmp_path: Path) -> None:
        vendor = _write_vendor(tmp_path)
        base_gguf = tmp_path / "base.fp16.gguf"
        base_gguf.write_bytes(b"ignored; base presence is all build_imatrix checks")
        export_dir = tmp_path / "exports" / "Q4_K_M"

        runner, calls = self._fake_runner()
        artifact = build_imatrix(
            base_gguf=base_gguf,
            calibration_text="some calibration text " * 50,
            export_dir=export_dir,
            base_revision="r1",
            corpus_sha256="c1",
            chunks=32,
            bin_override=vendor,
            subprocess_runner=runner,
        )
        assert len(calls) == 1
        assert artifact.path.is_file()
        assert (export_dir / "imatrix.meta.json").is_file()
        meta = json.loads((export_dir / "imatrix.meta.json").read_text())
        assert meta["base_revision"] == "r1"
        assert meta["corpus_sha256"] == "c1"
        assert meta["chunks"] == 32
        assert meta["sha256"] == artifact.sha256
        # Calib text file removed on success.
        assert not (export_dir / "imatrix.calib.txt").exists()

    def test_subprocess_leaves_calib_on_failure(self, tmp_path: Path) -> None:
        vendor = _write_vendor(tmp_path)
        base_gguf = tmp_path / "base.fp16.gguf"
        base_gguf.write_bytes(b"ok")
        export_dir = tmp_path / "exports" / "Q4_K_M"

        def runner(_argv: Any) -> None:
            raise SubprocessError(cmd=["x"], returncode=1, stderr_tail="boom")

        with pytest.raises(SubprocessError, match="boom"):
            build_imatrix(
                base_gguf=base_gguf,
                calibration_text="calibration text is here " * 20,
                export_dir=export_dir,
                base_revision="r1",
                corpus_sha256="c1",
                chunks=16,
                bin_override=vendor,
                subprocess_runner=runner,
            )
        # Operator should still be able to rerun the command by hand.
        assert (export_dir / "imatrix.calib.txt").is_file()

    def test_missing_output_raises_subprocess_error(self, tmp_path: Path) -> None:
        vendor = _write_vendor(tmp_path)
        base_gguf = tmp_path / "base.fp16.gguf"
        base_gguf.write_bytes(b"ok")
        export_dir = tmp_path / "exports" / "Q4_K_M"

        runner, _ = self._fake_runner(write_output=False)
        with pytest.raises(SubprocessError, match="not produced"):
            build_imatrix(
                base_gguf=base_gguf,
                calibration_text="x " * 100,
                export_dir=export_dir,
                base_revision="r1",
                corpus_sha256="c1",
                chunks=16,
                bin_override=vendor,
                subprocess_runner=runner,
            )

    def test_missing_base_gguf_raises(self, tmp_path: Path) -> None:
        vendor = _write_vendor(tmp_path)
        with pytest.raises(FileNotFoundError, match="imatrix base model missing"):
            build_imatrix(
                base_gguf=tmp_path / "nope.gguf",
                calibration_text="text",
                export_dir=tmp_path / "out",
                base_revision="r",
                corpus_sha256="c",
                bin_override=vendor,
                subprocess_runner=lambda _a: None,
            )

    @pytest.mark.parametrize(
        ("chunks", "chunk_size"),
        [(0, 512), (-1, 512), (256, 0), (256, -1)],
    )
    def test_nonpositive_params_raise(self, tmp_path: Path, chunks: int, chunk_size: int) -> None:
        vendor = _write_vendor(tmp_path)
        base = tmp_path / "b.gguf"
        base.write_bytes(b"x")
        with pytest.raises(ValueError):
            build_imatrix(
                base_gguf=base,
                calibration_text="text",
                export_dir=tmp_path / "out",
                base_revision="r",
                corpus_sha256="c",
                chunks=chunks,
                chunk_size=chunk_size,
                bin_override=vendor,
                subprocess_runner=lambda _a: None,
            )

    def test_empty_calibration_text_raises(self, tmp_path: Path) -> None:
        vendor = _write_vendor(tmp_path)
        base = tmp_path / "b.gguf"
        base.write_bytes(b"x")
        with pytest.raises(ValueError, match="calibration_text is empty"):
            build_imatrix(
                base_gguf=base,
                calibration_text="   \n\n\t  ",
                export_dir=tmp_path / "out",
                base_revision="r",
                corpus_sha256="c",
                bin_override=vendor,
                subprocess_runner=lambda _a: None,
            )


# --- resolve_imatrix ----------------------------------------------------------


class TestResolveImatrix:
    def _seed(
        self,
        tmp_path: Path,
        *,
        base_revision: str = "r1",
        corpus_sha256: str = "c1",
        chunks: int = DEFAULT_CHUNKS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        mutate_binary: bool = False,
    ) -> Path:
        export_dir = tmp_path / "exports" / "Q4_K_M"
        export_dir.mkdir(parents=True)
        binary = export_dir / "imatrix.gguf"
        binary.write_bytes(b"fake imatrix bytes")
        sha = _sha256_of_file(binary)
        if mutate_binary:
            binary.write_bytes(b"bytes have been changed")
        meta = {
            "path": "imatrix.gguf",
            "sha256": sha,
            "base_revision": base_revision,
            "corpus_sha256": corpus_sha256,
            "chunks": chunks,
            "chunk_size": chunk_size,
            "built_at": datetime.now(UTC).replace(tzinfo=None, microsecond=0).isoformat(),
        }
        (export_dir / "imatrix.meta.json").write_text(json.dumps(meta))
        return export_dir

    def test_matching_key_returns_artifact(self, tmp_path: Path) -> None:
        export_dir = self._seed(tmp_path)
        artifact = resolve_imatrix(
            export_dir,
            base_revision="r1",
            corpus_sha256="c1",
            chunks=DEFAULT_CHUNKS,
        )
        assert isinstance(artifact, ImatrixArtifact)
        assert artifact.base_revision == "r1"
        assert artifact.corpus_sha256 == "c1"

    def test_missing_binary_returns_none(self, tmp_path: Path) -> None:
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        assert (
            resolve_imatrix(
                export_dir,
                base_revision="r",
                corpus_sha256="c",
                chunks=DEFAULT_CHUNKS,
            )
            is None
        )

    def test_mismatched_corpus_sha_returns_none(self, tmp_path: Path) -> None:
        export_dir = self._seed(tmp_path, corpus_sha256="c1")
        assert (
            resolve_imatrix(
                export_dir,
                base_revision="r1",
                corpus_sha256="different",
                chunks=DEFAULT_CHUNKS,
            )
            is None
        )

    def test_mismatched_chunks_returns_none(self, tmp_path: Path) -> None:
        export_dir = self._seed(tmp_path, chunks=256)
        assert (
            resolve_imatrix(
                export_dir,
                base_revision="r1",
                corpus_sha256="c1",
                chunks=128,  # was 256
            )
            is None
        )

    def test_mismatched_base_revision_returns_none(self, tmp_path: Path) -> None:
        export_dir = self._seed(tmp_path, base_revision="r1")
        assert (
            resolve_imatrix(
                export_dir,
                base_revision="r2",
                corpus_sha256="c1",
                chunks=DEFAULT_CHUNKS,
            )
            is None
        )

    def test_tampered_binary_returns_none(self, tmp_path: Path) -> None:
        """A stale sidecar + post-mutation binary must look like a miss."""
        export_dir = self._seed(tmp_path, mutate_binary=True)
        assert (
            resolve_imatrix(
                export_dir,
                base_revision="r1",
                corpus_sha256="c1",
                chunks=DEFAULT_CHUNKS,
            )
            is None
        )

    def test_malformed_sidecar_returns_none(self, tmp_path: Path) -> None:
        export_dir = self._seed(tmp_path)
        (export_dir / "imatrix.meta.json").write_text("not json")
        assert (
            resolve_imatrix(
                export_dir,
                base_revision="r1",
                corpus_sha256="c1",
                chunks=DEFAULT_CHUNKS,
            )
            is None
        )

    def test_sidecar_wrong_shape_returns_none(self, tmp_path: Path) -> None:
        export_dir = self._seed(tmp_path)
        (export_dir / "imatrix.meta.json").write_text(json.dumps(["not", "a", "dict"]))
        assert (
            resolve_imatrix(
                export_dir,
                base_revision="r1",
                corpus_sha256="c1",
                chunks=DEFAULT_CHUNKS,
            )
            is None
        )


# --- calibration_text_from_replay --------------------------------------------


class TestCalibrationTextFromReplay:
    def test_missing_corpus_returns_sentinel(self, tmp_path: Path) -> None:
        text, sha = calibration_text_from_replay(
            corpus_path=tmp_path / "nope.zst",
            index_path=tmp_path / "idx.json",
        )
        assert text == ""
        assert sha == "<no-corpus>"

    def test_missing_index_tracks_corpus_sha(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.zst"
        corpus.write_bytes(b"deliberately-not-zstd bytes")
        text, sha = calibration_text_from_replay(
            corpus_path=corpus, index_path=tmp_path / "idx.json"
        )
        assert text == ""
        # Even without an index, we record the binary sha so a later
        # index write triggers a rebuild.
        assert sha != "<no-corpus>"
        assert len(sha) == 64

    def test_full_round_trip(self, tmp_path: Path) -> None:
        """Write a real replay via ReplayStore, then round-trip it."""
        from datetime import UTC
        from datetime import datetime as _dt

        from dlm.replay.models import SectionSnapshot
        from dlm.replay.store import ReplayStore

        corpus = tmp_path / "corpus.zst"
        idx = tmp_path / "index.json"
        store = ReplayStore.at(corpus, idx)

        snaps = [
            SectionSnapshot(
                section_id=f"{i:016x}",
                section_type="prose",
                content=f"Snapshot {i} prose content with lorem ipsum words.",
                first_seen_at=_dt(2026, 4, 19, tzinfo=UTC).replace(tzinfo=None),
                last_seen_at=_dt(2026, 4, 19, tzinfo=UTC).replace(tzinfo=None),
            )
            for i in range(5)
        ]
        store.append_many(snaps)

        text, sha = calibration_text_from_replay(corpus_path=corpus, index_path=idx)
        assert "Snapshot 0 prose content" in text
        assert "Snapshot 4 prose content" in text
        assert len(sha) == 64

    def test_truncates_at_max_chars(self, tmp_path: Path) -> None:
        from datetime import UTC
        from datetime import datetime as _dt

        from dlm.replay.models import SectionSnapshot
        from dlm.replay.store import ReplayStore

        corpus = tmp_path / "corpus.zst"
        idx = tmp_path / "index.json"
        store = ReplayStore.at(corpus, idx)
        big_content = "word " * 1000  # 5000 chars each
        snaps = [
            SectionSnapshot(
                section_id=f"{i:016x}",
                section_type="prose",
                content=big_content,
                first_seen_at=_dt(2026, 4, 19, tzinfo=UTC).replace(tzinfo=None),
                last_seen_at=_dt(2026, 4, 19, tzinfo=UTC).replace(tzinfo=None),
            )
            for i in range(10)
        ]
        store.append_many(snaps)

        text, _sha = calibration_text_from_replay(
            corpus_path=corpus, index_path=idx, max_chars=8_000
        )
        # `max_chars` is the pre-joiner content budget; the `\n\n`
        # separator between snapshots adds a small constant overhead.
        assert len(text) <= 8_000 + 2 * 10  # 10 possible joiners
