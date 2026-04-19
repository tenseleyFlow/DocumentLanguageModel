"""DlmLock schema — construction, round-trip, field constraints."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from dlm.lock.schema import CURRENT_LOCK_VERSION, DlmLock


def _minimal_lock(**overrides: object) -> DlmLock:
    """Build a valid DlmLock with every required field populated."""
    base = {
        "created_at": datetime(2026, 4, 19, 12, 0, 0, tzinfo=UTC),
        "dlm_id": "01HZXYAAAAAAAAAAAAAAAAAA00",
        "dlm_sha256": "0" * 64,
        "base_model_revision": "abc123",
        "hardware_tier": "cpu",
        "seed": 42,
        "determinism_class": "best-effort",
        "last_run_id": 1,
    }
    base.update(overrides)
    return DlmLock(**base)  # type: ignore[arg-type]


class TestDlmLockShape:
    def test_default_lock_version_is_current(self) -> None:
        assert _minimal_lock().lock_version == CURRENT_LOCK_VERSION

    def test_frozen(self) -> None:
        """`model_config.frozen=True` — mutation must raise."""
        lock = _minimal_lock()
        with pytest.raises(ValidationError):
            lock.seed = 99  # type: ignore[misc]

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            DlmLock(  # type: ignore[call-arg]
                created_at=datetime.now(UTC),
                dlm_id="01HZXY",
                dlm_sha256="0" * 64,
                base_model_revision="r",
                hardware_tier="cpu",
                seed=0,
                determinism_class="advisory",
                last_run_id=1,
                who_let_this_in="drift",
            )

    def test_dlm_sha256_must_be_hex_64(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_lock(dlm_sha256="deadbeef")  # too short

    def test_hardware_tier_literal_is_enforced(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_lock(hardware_tier="tpu")  # type: ignore[arg-type]

    def test_determinism_class_literal_is_enforced(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_lock(determinism_class="perfect")  # type: ignore[arg-type]

    def test_last_run_id_ge_1(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_lock(last_run_id=0)

    def test_pinned_versions_defaults_empty(self) -> None:
        assert _minimal_lock().pinned_versions == {}


class TestRoundTrip:
    def test_json_round_trip_preserves_fields(self) -> None:
        original = _minimal_lock(
            pinned_versions={"torch": "2.5.1", "transformers": "4.45.2"},
            determinism_flags={"cublas_workspace": ":4096:8", "use_det_algos": True},
            cuda_version="12.1",
        )
        payload = original.model_dump(mode="json")
        restored = DlmLock.model_validate(payload)
        assert restored == original
