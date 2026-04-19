"""validate_lock — mode dispatch + action classification."""

from __future__ import annotations

from datetime import UTC, datetime

from dlm.lock.policy import Severity
from dlm.lock.schema import DlmLock
from dlm.lock.validator import validate_lock


def _lock(**overrides: object) -> DlmLock:
    base = {
        "created_at": datetime(2026, 4, 19, 12, 0, 0, tzinfo=UTC),
        "dlm_id": "01HZXY",
        "dlm_sha256": "a" * 64,
        "base_model_revision": "rev1",
        "hardware_tier": "cpu",
        "seed": 42,
        "determinism_class": "best-effort",
        "last_run_id": 1,
        "pinned_versions": {"torch": "2.5.1"},
    }
    base.update(overrides)
    return DlmLock(**base)  # type: ignore[arg-type]


class TestNoPriorLock:
    def test_fresh_store_proceeds_and_writes(self) -> None:
        decision = validate_lock(None, _lock())
        assert decision.action == "proceed"
        assert decision.should_write_lock is True
        assert decision.mismatches == []


class TestDefaultMode:
    def test_identical_locks_proceed(self) -> None:
        lock = _lock()
        decision = validate_lock(lock, lock)
        assert decision.action == "proceed"
        assert decision.should_write_lock is True

    def test_dlm_edit_alone_proceeds(self) -> None:
        prior = _lock()
        current = _lock(dlm_sha256="b" * 64)
        decision = validate_lock(prior, current)
        # `ALLOW`-only mismatches shouldn't trigger warnings.
        assert decision.action == "proceed"

    def test_minor_drift_emits_proceed_with_warnings(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1"})
        current = _lock(pinned_versions={"torch": "2.6.0"})
        decision = validate_lock(prior, current)
        assert decision.action == "proceed_with_warnings"
        assert decision.should_write_lock is True
        assert any(sev is Severity.WARN for sev, _ in decision.mismatches)

    def test_base_revision_change_aborts(self) -> None:
        prior = _lock(base_model_revision="rev1")
        current = _lock(base_model_revision="rev2")
        decision = validate_lock(prior, current)
        assert decision.action == "abort"
        assert decision.should_write_lock is False

    def test_torch_major_change_aborts(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1"})
        current = _lock(pinned_versions={"torch": "3.0.0"})
        assert validate_lock(prior, current).action == "abort"


class TestStrictMode:
    def test_warns_become_errors_and_abort(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1"})
        current = _lock(pinned_versions={"torch": "2.6.0"})
        decision = validate_lock(prior, current, mode="strict")
        assert decision.action == "abort"
        assert decision.should_write_lock is False

    def test_allows_still_allow(self) -> None:
        prior = _lock()
        current = _lock(dlm_sha256="b" * 64)
        decision = validate_lock(prior, current, mode="strict")
        assert decision.action == "proceed"


class TestUpdateAndIgnore:
    def test_update_skips_validation_and_writes(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1"})
        current = _lock(pinned_versions={"torch": "3.0.0"})
        decision = validate_lock(prior, current, mode="update")
        assert decision.action == "proceed"
        assert decision.should_write_lock is True
        assert decision.mismatches == []

    def test_ignore_skips_validation_and_does_not_write(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1"})
        current = _lock(pinned_versions={"torch": "3.0.0"})
        decision = validate_lock(prior, current, mode="ignore")
        assert decision.action == "proceed"
        assert decision.should_write_lock is False
