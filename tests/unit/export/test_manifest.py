"""ExportManifest — schema + round-trip + sha256 helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from dlm.export.errors import ExportManifestError
from dlm.export.manifest import (
    EXPORT_MANIFEST_FILENAME,
    ExportArtifact,
    ExportManifest,
    build_artifact,
    compute_sha256,
    load_export_manifest,
    save_export_manifest,
    utc_now,
)


def _manifest(**overrides: object) -> ExportManifest:
    base: dict[str, object] = {
        "quant": "Q4_K_M",
        "merged": False,
        "dequantized": False,
        "ollama_name": None,
        "created_at": datetime(2026, 4, 18, 22, 0, 0),
        "created_by": "dlm-0.1.0",
        "llama_cpp_tag": "b1234",
        "base_model_hf_id": "org/base",
        "base_model_revision": "a" * 40,
        "adapter_version": 1,
        "artifacts": [],
    }
    base.update(overrides)
    return ExportManifest.model_validate(base)


class TestComputeSha256:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        data = b"hello world" * 100
        path = tmp_path / "blob"
        path.write_bytes(data)
        assert compute_sha256(path) == hashlib.sha256(data).hexdigest()

    def test_streaming_larger_than_chunk(self, tmp_path: Path) -> None:
        data = b"x" * (2 * (1 << 20) + 7)  # 2MB + tail
        path = tmp_path / "blob"
        path.write_bytes(data)
        assert compute_sha256(path) == hashlib.sha256(data).hexdigest()


class TestBuildArtifact:
    def test_relative_path_computed(self, tmp_path: Path) -> None:
        subdir = tmp_path / "exports" / "Q4_K_M"
        subdir.mkdir(parents=True)
        f = subdir / "adapter.gguf"
        f.write_bytes(b"hello")
        art = build_artifact(subdir, f)
        assert art.path == "adapter.gguf"
        assert art.size_bytes == 5
        assert art.sha256 == hashlib.sha256(b"hello").hexdigest()


class TestSchema:
    def test_minimal_valid(self) -> None:
        m = _manifest()
        assert m.quant == "Q4_K_M"
        assert m.llama_cpp_tag == "b1234"

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ExportManifest.model_validate(
                {
                    **_manifest().model_dump(mode="json"),
                    "surprise": "nope",
                }
            )

    def test_sha_length_enforced(self) -> None:
        with pytest.raises(ValidationError):
            ExportArtifact(path="x", sha256="short", size_bytes=0)


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        m = _manifest(
            artifacts=[
                ExportArtifact(
                    path="base.Q4_K_M.gguf",
                    sha256="a" * 64,
                    size_bytes=1_000_000,
                )
            ]
        )
        saved = save_export_manifest(tmp_path, m)
        assert saved.name == EXPORT_MANIFEST_FILENAME
        back = load_export_manifest(tmp_path)
        assert back == m

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ExportManifestError, match="missing"):
            load_export_manifest(tmp_path)

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / EXPORT_MANIFEST_FILENAME).write_text("not json {{{")
        with pytest.raises(ExportManifestError, match="cannot parse"):
            load_export_manifest(tmp_path)

    def test_bad_shape_raises(self, tmp_path: Path) -> None:
        (tmp_path / EXPORT_MANIFEST_FILENAME).write_text(json.dumps({"quant": "Q4_K_M"}))
        with pytest.raises(ExportManifestError, match="invalid shape"):
            load_export_manifest(tmp_path)

    def test_sorted_json_on_disk(self, tmp_path: Path) -> None:
        save_export_manifest(tmp_path, _manifest())
        text = (tmp_path / EXPORT_MANIFEST_FILENAME).read_text()
        data = json.loads(text)
        assert list(data.keys()) == sorted(data.keys())


class TestUtcNow:
    def test_tz_naive_microsecond_zero(self) -> None:
        now = utc_now()
        assert now.tzinfo is None
        assert now.microsecond == 0
