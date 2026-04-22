"""training_state.pt save/load/integrity/version-drift."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from dlm.train.errors import ResumeIntegrityError, VersionDriftWarning
from dlm.train.state_sidecar import (
    RNG_SIDECAR_FILENAME,
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


class TestRngSidecar:
    """Audit-11 B7: numpy + python RNG round-trip via JSON sidecar.

    The v1 layout stored these inside the torch payload and required
    `weights_only=False` on load — an RCE vector. v2 moves them to a
    JSON sidecar so the torch payload loads safely under
    `weights_only=True`.
    """

    def test_numpy_rng_round_trip(self, tmp_path: Path) -> None:
        """`numpy.random.get_state()` must round-trip exactly — any drift
        means loss curves diverge on resume even with matched torch RNG."""
        rng = np.random.RandomState(seed=12345)
        rng.random_sample(100)  # advance the state past seed init
        original_state: tuple[Any, ...] = rng.get_state(legacy=True)  # type: ignore[assignment]

        state = _mock_state()
        state["numpy_rng_state"] = original_state
        save_state(tmp_path, state)

        loaded = load_state(tmp_path, runtime_versions={"torch": torch.__version__})
        restored: tuple[Any, ...] = loaded["numpy_rng_state"]

        assert restored[0] == original_state[0]
        np.testing.assert_array_equal(restored[1], original_state[1])
        assert restored[2] == original_state[2]
        assert restored[3] == original_state[3]
        assert restored[4] == original_state[4]

    def test_numpy_rng_round_trip_draws_match(self, tmp_path: Path) -> None:
        """Behavioral check: after `set_state(restored)`, the next draws
        match what the original generator would have produced."""
        rng = np.random.RandomState(seed=99)
        rng.random_sample(50)
        original_state = rng.get_state()
        expected_draws = rng.random_sample(20)

        state = _mock_state()
        state["numpy_rng_state"] = original_state
        save_state(tmp_path, state)

        loaded = load_state(tmp_path, runtime_versions={"torch": torch.__version__})
        resumed = np.random.RandomState()
        resumed.set_state(loaded["numpy_rng_state"])
        np.testing.assert_array_equal(resumed.random_sample(20), expected_draws)

    def test_python_random_round_trip(self, tmp_path: Path) -> None:
        """`random.getstate()` must round-trip so replay sampling matches."""
        rng = random.Random(7)
        for _ in range(50):
            rng.random()
        original_state = rng.getstate()
        expected = [rng.random() for _ in range(10)]

        state = _mock_state()
        state["python_random_state"] = original_state
        save_state(tmp_path, state)

        loaded = load_state(tmp_path, runtime_versions={"torch": torch.__version__})
        resumed = random.Random()
        resumed.setstate(loaded["python_random_state"])
        got = [resumed.random() for _ in range(10)]
        assert got == expected

    def test_rng_sidecar_is_valid_json(self, tmp_path: Path) -> None:
        """The sidecar is parseable JSON with the expected top-level keys."""
        state = _mock_state()
        state["numpy_rng_state"] = np.random.RandomState(seed=1).get_state()
        save_state(tmp_path, state)

        sidecar = json.loads((tmp_path / RNG_SIDECAR_FILENAME).read_text())
        assert sidecar["_rng_sidecar_version"] == 2
        assert sidecar["numpy_rng_state"] is not None
        assert "state_hex" in sidecar["numpy_rng_state"]
        assert sidecar["python_random_state"] is not None

    def test_missing_rng_sidecar_on_v2_payload_raises(self, tmp_path: Path) -> None:
        """Deleting `training_state.rng.json` after a v2 save must be
        refused — silently substituting None breaks determinism."""
        save_state(tmp_path, _mock_state())
        (tmp_path / RNG_SIDECAR_FILENAME).unlink()

        with pytest.raises(ResumeIntegrityError, match="requires training_state.rng.json"):
            load_state(tmp_path, runtime_versions={"torch": torch.__version__})

    def test_malformed_rng_sidecar_raises(self, tmp_path: Path) -> None:
        save_state(tmp_path, _mock_state())
        (tmp_path / RNG_SIDECAR_FILENAME).write_text("{not valid json")

        with pytest.raises(ResumeIntegrityError, match="cannot read RNG sidecar"):
            load_state(tmp_path, runtime_versions={"torch": torch.__version__})

    def test_torch_payload_loads_under_weights_only_true(self, tmp_path: Path) -> None:
        """Direct verification the payload never needs `weights_only=False`.

        The point of the v2 refactor: tampered pickled bytes cannot
        execute arbitrary code on resume because the loader itself
        refuses to deserialize non-allowlisted types.
        """
        state = _mock_state()
        state["numpy_rng_state"] = np.random.RandomState(seed=1).get_state()
        save_state(tmp_path, state)

        blob = (tmp_path / STATE_FILENAME).read_bytes()
        payload = torch.load(io.BytesIO(blob), weights_only=True)
        assert payload["_state_sidecar_version"] == 2
        # numpy ndarrays shouldn't appear in the torch payload — they
        # live in the JSON sidecar.
        assert "numpy_rng_state" not in payload


class TestLegacyV1Compat:
    """Audit-11 B7: one-release back-compat for pre-B7 sidecars.

    Prior releases torch.save'd the full state dict (including numpy
    ndarrays) under `weights_only=False`. v2's reader retries with the
    legacy loader + logs a migration warning so existing checkpoints
    keep resuming through the transition.
    """

    def _write_v1_sidecar(self, directory: Path) -> None:
        """Emit a v1-shape blob (no version marker, numpy array inline)."""
        payload = {
            "optimizer_state_dict": {"lr": 1e-4},
            "scheduler_state_dict": {"step": 5},
            "scaler_state_dict": None,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": None,
            "numpy_rng_state": np.random.RandomState(seed=1).get_state(),
            "python_random_state": random.getstate(),
            "global_step": 10,
            "epoch": 0.5,
            "best_val_loss": 0.9,
            "dlm_manifest_hash": None,
            "base_model_revision": "a" * 40,
            "pinned_versions": {"torch": torch.__version__},
            "use_qlora": False,
        }
        buf = io.BytesIO()
        torch.save(payload, buf)
        blob = buf.getvalue()
        (directory / STATE_FILENAME).write_bytes(blob)
        (directory / STATE_SHA_FILENAME).write_text(hashlib.sha256(blob).hexdigest() + "\n")

    def test_v1_payload_loads_with_migration_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._write_v1_sidecar(tmp_path)

        with caplog.at_level(logging.WARNING, logger="dlm.train.state_sidecar"):
            loaded = load_state(tmp_path, runtime_versions={"torch": torch.__version__})

        assert loaded["global_step"] == 10
        assert loaded["numpy_rng_state"] is not None
        assert any("legacy v1 format" in rec.message for rec in caplog.records)

    def test_v1_payload_does_not_require_rng_sidecar(self, tmp_path: Path) -> None:
        """The legacy path carries RNG inline, so the JSON sidecar
        must not be required for v1 blobs."""
        self._write_v1_sidecar(tmp_path)
        # No RNG_SIDECAR_FILENAME written — this must still load.
        loaded = load_state(tmp_path, runtime_versions={"torch": torch.__version__})
        assert loaded["global_step"] == 10


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
