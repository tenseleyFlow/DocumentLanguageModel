"""builder.py — hash helper + hardware-tier mapping + DlmLock assembly."""

from __future__ import annotations

import hashlib
from pathlib import Path

from dlm.lock.builder import build_lock, hardware_tier_from_backend, hash_dlm_file
from dlm.lock.schema import CURRENT_LOCK_VERSION


class TestHashDlmFile:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.dlm"
        content = b"hello, world\n"
        path.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert hash_dlm_file(path) == expected

    def test_streams_large_file(self, tmp_path: Path) -> None:
        path = tmp_path / "big.dlm"
        path.write_bytes(b"A" * (200_000))  # > 65 KiB chunk boundary
        assert hash_dlm_file(path) == hashlib.sha256(b"A" * 200_000).hexdigest()


class TestHardwareTierFromBackend:
    def test_cuda_sm_ge_80_is_modern_tier(self) -> None:
        assert hardware_tier_from_backend("cuda", sm=(8, 0)) == "cuda-sm80+"
        assert hardware_tier_from_backend("cuda", sm=(9, 0)) == "cuda-sm80+"

    def test_cuda_sm_lt_80_is_legacy_tier(self) -> None:
        assert hardware_tier_from_backend("cuda", sm=(7, 5)) == "cuda-sm<80"

    def test_cuda_without_sm_falls_back_to_legacy(self) -> None:
        assert hardware_tier_from_backend("cuda", sm=None) == "cuda-sm<80"

    def test_mps_rocm_cpu_pass_through(self) -> None:
        assert hardware_tier_from_backend("mps") == "mps"
        assert hardware_tier_from_backend("rocm") == "rocm"
        assert hardware_tier_from_backend("cpu") == "cpu"

    def test_unknown_backend_is_cpu(self) -> None:
        assert hardware_tier_from_backend(None) == "cpu"
        assert hardware_tier_from_backend("xla") == "cpu"


class TestBuildLock:
    def test_minimum_args_populate_required_fields(self) -> None:
        lock = build_lock(
            dlm_id="01HZ",
            dlm_sha256="a" * 64,
            base_model_revision="rev",
            hardware_tier="cpu",
            seed=42,
            determinism_class="best-effort",
            run_id=1,
        )
        assert lock.lock_version == CURRENT_LOCK_VERSION
        assert lock.last_run_id == 1
        assert lock.pinned_versions == {}
        assert lock.determinism_flags == {}
        assert lock.license_acceptance is None

    def test_optional_fields_pass_through(self) -> None:
        lock = build_lock(
            dlm_id="01HZ",
            dlm_sha256="a" * 64,
            base_model_revision="rev",
            hardware_tier="cuda-sm80+",
            seed=7,
            determinism_class="strong",
            run_id=5,
            pinned_versions={"torch": "2.5.1"},
            cuda_version="12.1",
            determinism_flags={"cublas_workspace": ":4096:8"},
        )
        assert lock.cuda_version == "12.1"
        assert lock.pinned_versions == {"torch": "2.5.1"}
        assert lock.determinism_flags == {"cublas_workspace": ":4096:8"}
