"""Downloader contract — pinned revision, directory sha256, error paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dlm.base_models import BaseModelSpec, GatedModelError, download_spec, sha256_of_directory


def _spec() -> BaseModelSpec:
    return BaseModelSpec.model_validate(
        {
            "key": "demo-1b",
            "hf_id": "org/demo",
            "revision": "a" * 40,
            "architecture": "DemoForCausalLM",
            "params": 1_000_000_000,
            "target_modules": ["q_proj", "v_proj"],
            "template": "chatml",
            "gguf_arch": "demo",
            "tokenizer_pre": "demo",
            "license_spdx": "MIT",
            "requires_acceptance": False,
            "redistributable": True,
            "size_gb_fp16": 2.0,
            "context_length": 4096,
            "recommended_seq_len": 2048,
        }
    )


class TestDirectorySha256:
    def test_same_contents_same_digest(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        for root in (a, b):
            root.mkdir()
            (root / "config.json").write_bytes(b'{"x": 1}')
            (root / "model.safetensors").write_bytes(b"\x00" * 64)
        assert sha256_of_directory(a) == sha256_of_directory(b)

    def test_different_content_different_digest(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "config.json").write_bytes(b'{"x": 1}')
        (b / "config.json").write_bytes(b'{"x": 2}')
        assert sha256_of_directory(a) != sha256_of_directory(b)

    def test_different_paths_different_digest(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "config.json").write_bytes(b"content")
        (b / "other.json").write_bytes(b"content")
        assert sha256_of_directory(a) != sha256_of_directory(b)

    def test_deterministic_across_runs(self, tmp_path: Path) -> None:
        """Same tree = same digest, invoked twice."""
        root = tmp_path / "r"
        root.mkdir()
        (root / "a.txt").write_bytes(b"hello")
        (root / "b.txt").write_bytes(b"world")
        first = sha256_of_directory(root)
        second = sha256_of_directory(root)
        assert first == second

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            sha256_of_directory(tmp_path / "absent")

    def test_only_files_included_not_dirs(self, tmp_path: Path) -> None:
        """Empty subdirectories don't affect the digest."""
        root = tmp_path / "r"
        root.mkdir()
        (root / "file.txt").write_bytes(b"hi")
        without_empty = sha256_of_directory(root)
        (root / "empty-dir").mkdir()
        with_empty = sha256_of_directory(root)
        assert without_empty == with_empty


class TestDownloadSpec:
    def test_returns_result_with_pinned_revision(self, tmp_path: Path) -> None:
        spec = _spec()
        snapshot = tmp_path / "hub" / "models--org--demo" / "snapshots" / spec.revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_bytes(b'{"arch":"demo"}')

        with patch("huggingface_hub.snapshot_download", return_value=str(snapshot)):
            result = download_spec(spec)
        assert result.path == snapshot
        assert result.revision == spec.revision
        # Digest matches our standalone computation
        expected_digest = sha256_of_directory(snapshot)
        assert result.sha256 == expected_digest

    def test_revision_mismatch_raises(self, tmp_path: Path) -> None:
        spec = _spec()
        # Snapshot under a DIFFERENT sha to simulate a revision race.
        other_sha = "b" * 40
        snapshot = tmp_path / "hub" / "models--org--demo" / "snapshots" / other_sha
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_bytes(b"{}")

        with (
            patch("huggingface_hub.snapshot_download", return_value=str(snapshot)),
            pytest.raises(RuntimeError, match="revision mismatch"),
        ):
            download_spec(spec)

    def test_gated_repo_surfaces_as_gated_model_error(self) -> None:
        from unittest.mock import Mock

        from huggingface_hub.errors import GatedRepoError

        with (
            patch(
                "huggingface_hub.snapshot_download",
                side_effect=GatedRepoError("gated", response=Mock()),
            ),
            pytest.raises(GatedModelError),
        ):
            download_spec(_spec())

    def test_local_files_only_refuses_when_absent(self) -> None:
        from huggingface_hub.errors import LocalEntryNotFoundError

        with (
            patch(
                "huggingface_hub.snapshot_download",
                side_effect=LocalEntryNotFoundError("not cached"),
            ),
            pytest.raises(RuntimeError, match="offline"),
        ):
            download_spec(_spec(), local_files_only=True)
