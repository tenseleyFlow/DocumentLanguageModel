"""Severity table — every row in the markdown policy has coverage here."""

from __future__ import annotations

from datetime import UTC, datetime

from dlm.lock.policy import Severity, classify_mismatches
from dlm.lock.schema import DlmLock


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
        "pinned_versions": {"torch": "2.5.1", "transformers": "4.45.2"},
        "determinism_flags": {},
    }
    base.update(overrides)
    return DlmLock(**base)  # type: ignore[arg-type]


def _severities(prior: DlmLock, current: DlmLock, *, strict: bool = False) -> set[Severity]:
    return {sev for sev, _msg in classify_mismatches(prior, current, strict=strict)}


class TestDlmShaAlwaysAllowed:
    def test_dlm_sha_change_is_allow(self) -> None:
        prior = _lock()
        current = _lock(dlm_sha256="b" * 64)
        assert _severities(prior, current) == {Severity.ALLOW}

    def test_dlm_sha_change_still_allow_under_strict(self) -> None:
        prior = _lock()
        current = _lock(dlm_sha256="b" * 64)
        assert _severities(prior, current, strict=True) == {Severity.ALLOW}


class TestBaseRevision:
    def test_base_revision_change_is_error(self) -> None:
        prior = _lock()
        current = _lock(base_model_revision="rev2")
        assert Severity.ERROR in _severities(prior, current)


class TestTorchVersion:
    def test_major_version_change_is_error(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1"})
        current = _lock(pinned_versions={"torch": "3.0.0"})
        assert Severity.ERROR in _severities(prior, current)

    def test_minor_version_change_is_warn(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1"})
        current = _lock(pinned_versions={"torch": "2.6.0"})
        outcomes = classify_mismatches(prior, current)
        torch_outcomes = [x for x in outcomes if "torch" in x[1]]
        assert torch_outcomes == [
            (Severity.WARN, "torch minor-version drift (2.5.1 → 2.6.0)"),
        ]

    def test_strict_upgrades_minor_warn_to_error(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1"})
        current = _lock(pinned_versions={"torch": "2.6.0"})
        assert Severity.ERROR in _severities(prior, current, strict=True)


class TestMinorPeers:
    def test_transformers_change_is_warn(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1", "transformers": "4.45.2"})
        current = _lock(pinned_versions={"torch": "2.5.1", "transformers": "4.46.0"})
        assert _severities(prior, current) == {Severity.WARN}

    def test_peft_trl_accelerate_llama_cpp_each_fire(self) -> None:
        for key, a, b in [
            ("peft", "0.13.0", "0.14.0"),
            ("trl", "0.11.4", "0.12.0"),
            ("accelerate", "0.35.0", "0.36.0"),
            ("llama_cpp", "b8816", "b9000"),
        ]:
            prior = _lock(pinned_versions={"torch": "2.5.1", key: a})
            current = _lock(pinned_versions={"torch": "2.5.1", key: b})
            msgs = [msg for _, msg in classify_mismatches(prior, current) if key in msg]
            assert msgs, f"expected {key} drift to surface"


class TestBitsAndBytes:
    def test_bnb_drift_is_warn(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1", "bitsandbytes": "0.43.0"})
        current = _lock(pinned_versions={"torch": "2.5.1", "bitsandbytes": "0.43.1"})
        msgs = [msg for _, msg in classify_mismatches(prior, current) if "bitsandbytes" in msg]
        assert msgs
        assert _severities(prior, current) == {Severity.WARN}

    def test_bnb_added_or_removed_is_warn(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1"})
        current = _lock(pinned_versions={"torch": "2.5.1", "bitsandbytes": "0.43.0"})
        assert _severities(prior, current) == {Severity.WARN}


class TestHardwareAndDeterminism:
    def test_hardware_tier_change_is_warn(self) -> None:
        prior = _lock(hardware_tier="cpu")
        current = _lock(hardware_tier="mps")
        assert _severities(prior, current) == {Severity.WARN}

    def test_determinism_class_change_is_warn(self) -> None:
        prior = _lock(determinism_class="strong")
        current = _lock(determinism_class="best-effort")
        assert _severities(prior, current) == {Severity.WARN}

    def test_determinism_flags_change_is_warn(self) -> None:
        prior = _lock(determinism_flags={"cublas": ":4096:8"})
        current = _lock(determinism_flags={"cublas": ":16:8"})
        assert _severities(prior, current) == {Severity.WARN}


class TestNoDrift:
    def test_identical_locks_produce_no_mismatches(self) -> None:
        lock = _lock()
        assert classify_mismatches(lock, lock) == []
