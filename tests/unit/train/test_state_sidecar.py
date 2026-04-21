"""training_state.pt save/load/integrity/version-drift."""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import pytest
import torch

from dlm.train.errors import ResumeIntegrityError, VersionDriftWarning
from dlm.train.state_sidecar import (
    STATE_FILENAME,
    STATE_SHA_FILENAME,
    TRAINING_RUN_FILENAME,
    VERSIONS_FILENAME,
    TrainingState,
    capture_runtime_versions,
    load_state,
    save_state,
)


def _mock_state(*, use_qlora: bool = False) -> TrainingState:
    return TrainingState(
        optimizer_state_dict={"lr": 1e-4},
        scheduler_state_dict={"step": 5},
        scaler_state_dict=None,
        torch_rng_state=torch.get_rng_state(),
        cuda_rng_state=None,
        numpy_rng_state=None,
        python_random_state=random.getstate(),
        global_step=10,
        epoch=0.5,
        best_val_loss=0.9,
        dlm_manifest_hash=None,
        base_model_revision="a" * 40,
        pinned_versions={"torch": torch.__version__},
        use_qlora=use_qlora,
    )


class TestRoundTrip:
    def test_save_writes_three_files(self, tmp_path: Path) -> None:
        save_state(tmp_path, _mock_state())
        assert (tmp_path / STATE_FILENAME).exists()
        assert (tmp_path / STATE_SHA_FILENAME).exists()
        assert (tmp_path / VERSIONS_FILENAME).exists()

    def test_load_returns_state(self, tmp_path: Path) -> None:
        save_state(tmp_path, _mock_state())
        loaded = load_state(tmp_path, runtime_versions={"torch": torch.__version__})
        assert loaded["global_step"] == 10
        assert loaded["base_model_revision"] == "a" * 40

    def test_pinned_versions_json_sidecar_readable(self, tmp_path: Path) -> None:
        """`pinned_versions.json` is JSON for human grep-ability."""
        import json

        save_state(tmp_path, _mock_state())
        content = json.loads((tmp_path / VERSIONS_FILENAME).read_text())
        assert "torch" in content


class TestIntegrity:
    def test_missing_state_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ResumeIntegrityError, match="missing training state"):
            load_state(tmp_path, runtime_versions={})

    def test_missing_sha_file_raises(self, tmp_path: Path) -> None:
        save_state(tmp_path, _mock_state())
        (tmp_path / STATE_SHA_FILENAME).unlink()
        with pytest.raises(ResumeIntegrityError, match="sha256 sidecar"):
            load_state(tmp_path, runtime_versions={})

    def test_corrupted_state_raises(self, tmp_path: Path) -> None:
        save_state(tmp_path, _mock_state())
        (tmp_path / STATE_FILENAME).write_bytes(b"tampered-bytes")
        with pytest.raises(ResumeIntegrityError, match="sha256 mismatch"):
            load_state(tmp_path, runtime_versions={})

    def test_corrupted_sha_raises(self, tmp_path: Path) -> None:
        save_state(tmp_path, _mock_state())
        (tmp_path / STATE_SHA_FILENAME).write_text("0" * 64 + "\n")
        with pytest.raises(ResumeIntegrityError, match="sha256 mismatch"):
            load_state(tmp_path, runtime_versions={})


class TestVersionDrift:
    def test_matching_versions_no_warning(self, tmp_path: Path) -> None:
        save_state(tmp_path, _mock_state())
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails
            load_state(tmp_path, runtime_versions={"torch": torch.__version__})

    def test_differing_version_emits_warning(self, tmp_path: Path) -> None:
        save_state(tmp_path, _mock_state())
        with pytest.warns(VersionDriftWarning, match="torch:"):
            load_state(tmp_path, runtime_versions={"torch": "99.99.99"})

    def test_gaining_a_package_is_not_drift(self, tmp_path: Path) -> None:
        """Saved had no `trl` pinned; current runtime knows it → no drift.

        Gaining capability isn't drift — there was no prior state to
        diverge from. Only losing a pinned package is (see M6 test).
        """
        save_state(tmp_path, _mock_state())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_state(
                tmp_path,
                runtime_versions={"torch": torch.__version__, "trl": "1.2.0"},
            )

    def test_losing_pinned_package_is_drift(self, tmp_path: Path) -> None:
        """Audit-04 M6: saved had `bitsandbytes="0.43.1"`, runtime has None.

        This matters for the QLoRA-on-CUDA → resumed-on-Apple-Silicon
        case; under the old logic it was silently skipped.
        """
        # Build a mock state whose pinned_versions declares bitsandbytes.
        state = _mock_state()
        state["pinned_versions"] = {
            "torch": torch.__version__,
            "bitsandbytes": "0.43.1",
        }
        save_state(tmp_path, state)

        with pytest.warns(VersionDriftWarning, match="bitsandbytes.*0\\.43\\.1.*missing"):
            load_state(
                tmp_path,
                runtime_versions={"torch": torch.__version__, "bitsandbytes": None},
            )

    def test_losing_pinned_package_missing_key_is_drift(self, tmp_path: Path) -> None:
        """Same as above but the runtime dict omits the key entirely."""
        state = _mock_state()
        state["pinned_versions"] = {"torch": torch.__version__, "bitsandbytes": "0.43.1"}
        save_state(tmp_path, state)

        with pytest.warns(VersionDriftWarning, match="bitsandbytes.*missing"):
            load_state(tmp_path, runtime_versions={"torch": torch.__version__})


class TestTrainingRunSidecar:
    """Audit-05 M1: explicit use_qlora flag persisted alongside the adapter."""

    def test_training_run_json_written_on_save(self, tmp_path: Path) -> None:
        import json

        save_state(tmp_path, _mock_state(use_qlora=True))
        training_run = tmp_path / TRAINING_RUN_FILENAME
        assert training_run.exists()
        data = json.loads(training_run.read_text())
        assert data["use_qlora"] is True

    def test_training_run_defaults_false_when_lora(self, tmp_path: Path) -> None:
        import json

        save_state(tmp_path, _mock_state(use_qlora=False))
        data = json.loads((tmp_path / TRAINING_RUN_FILENAME).read_text())
        assert data["use_qlora"] is False


class TestCaptureRuntimeVersions:
    def test_torch_key_populated(self) -> None:
        versions = capture_runtime_versions()
        assert "torch" in versions
        assert isinstance(versions["torch"], str)

    def test_bitsandbytes_key_present_even_if_none(self) -> None:
        """Explicit `None` so the key survives a JSON round-trip + drift check."""
        versions = capture_runtime_versions()
        # Value may be None on Apple Silicon (bnb not installed) — but key exists.
        assert "bitsandbytes" in versions

    def test_sway_key_present_even_if_none(self) -> None:
        """Same shape as bitsandbytes: key always present, `None` when sway is
        not installed in this venv. Records which probe harness produced the
        reports that drove the run."""
        versions = capture_runtime_versions()
        assert "sway" in versions
