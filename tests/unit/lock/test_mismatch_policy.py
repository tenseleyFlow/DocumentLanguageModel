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


class TestWorldSize:
    """Audit-08 B1: world_size drift must fire WARN."""

    def test_world_size_match_is_allow(self) -> None:
        prior = _lock(world_size=1)
        current = _lock(world_size=1)
        assert Severity.WARN not in _severities(prior, current)

    def test_world_size_increase_is_warn(self) -> None:
        prior = _lock(world_size=1)
        current = _lock(world_size=4)
        sevs = _severities(prior, current)
        assert Severity.WARN in sevs
        # Surface message mentions both numbers + the drift rationale.
        msgs = [m for _s, m in classify_mismatches(prior, current)]
        assert any("world_size changed (1 → 4)" in m for m in msgs)

    def test_world_size_decrease_is_warn(self) -> None:
        prior = _lock(world_size=8)
        current = _lock(world_size=1)
        assert Severity.WARN in _severities(prior, current)

    def test_world_size_strict_upgrades_to_error(self) -> None:
        prior = _lock(world_size=2)
        current = _lock(world_size=4)
        assert Severity.ERROR in _severities(prior, current, strict=True)


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


class TestNoneTransitions:
    """Audit-05 M3: torch / bnb / minor-peers rules must surface
    one-sided None transitions (version added / removed) as WARN rather
    than silently allowing them."""

    def test_torch_added_is_warn(self) -> None:
        prior = _lock(pinned_versions={})
        current = _lock(pinned_versions={"torch": "2.5.1"})
        assert _severities(prior, current) == {Severity.WARN}

    def test_torch_removed_is_warn(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1"})
        current = _lock(pinned_versions={})
        assert _severities(prior, current) == {Severity.WARN}

    def test_transformers_added_is_warn(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1"})
        current = _lock(pinned_versions={"torch": "2.5.1", "transformers": "4.46.0"})
        assert _severities(prior, current) == {Severity.WARN}

    def test_bitsandbytes_removed_is_warn(self) -> None:
        prior = _lock(pinned_versions={"torch": "2.5.1", "bitsandbytes": "0.43.0"})
        current = _lock(pinned_versions={"torch": "2.5.1"})
        assert _severities(prior, current) == {Severity.WARN}


class TestSeedRule:
    def test_seed_change_is_warn(self) -> None:
        prior = _lock(seed=42)
        current = _lock(seed=99)
        assert _severities(prior, current) == {Severity.WARN}

    def test_seed_change_upgrades_under_strict(self) -> None:
        prior = _lock(seed=42)
        current = _lock(seed=99)
        assert _severities(prior, current, strict=True) == {Severity.ERROR}


class TestBaseModelSha256Rule:
    def test_sha256_mismatch_is_error(self) -> None:
        prior = _lock(base_model_sha256="a" * 64)
        current = _lock(base_model_sha256="b" * 64)
        assert Severity.ERROR in _severities(prior, current)

    def test_one_sided_none_is_silent(self) -> None:
        prior = _lock(base_model_sha256=None)
        current = _lock(base_model_sha256="a" * 64)
        # No drift under the sha256 rule when the prior didn't capture it.
        messages = [msg for _sev, msg in classify_mismatches(prior, current)]
        assert not any("base_model_sha256" in m for m in messages)


class TestCudaRocmRules:
    def test_cuda_version_change_is_warn(self) -> None:
        prior = _lock(cuda_version="12.1")
        current = _lock(cuda_version="12.4")
        assert _severities(prior, current) == {Severity.WARN}

    def test_rocm_version_change_is_warn(self) -> None:
        prior = _lock(rocm_version="6.0")
        current = _lock(rocm_version="6.2")
        assert _severities(prior, current) == {Severity.WARN}


class TestLicenseAcceptanceRule:
    def _acceptance(self, spdx: str = "llama3.2", url: str = "https://example.com/llama") -> object:
        from dlm.base_models.license import LicenseAcceptance

        return LicenseAcceptance(
            accepted_at=datetime(2026, 4, 1, tzinfo=UTC),
            license_url=url,
            license_spdx=spdx,
            via="cli_flag",
        )

    def test_none_to_populated_is_warn(self) -> None:
        prior = _lock(license_acceptance=None)
        current = _lock(license_acceptance=self._acceptance())
        msgs = [msg for _s, msg in classify_mismatches(prior, current)]
        assert any("license_acceptance newly recorded" in m for m in msgs)

    def test_spdx_change_is_warn(self) -> None:
        prior = _lock(license_acceptance=self._acceptance(spdx="llama3.2"))
        current = _lock(license_acceptance=self._acceptance(spdx="llama3.3"))
        msgs = [msg for _s, msg in classify_mismatches(prior, current)]
        assert any("spdx changed" in m for m in msgs)

    def test_both_none_is_silent(self) -> None:
        prior = _lock(license_acceptance=None)
        current = _lock(license_acceptance=None)
        msgs = [msg for _s, msg in classify_mismatches(prior, current)]
        assert not any("license_acceptance" in m for m in msgs)


class TestNoDrift:
    def test_identical_locks_produce_no_mismatches(self) -> None:
        lock = _lock()
        assert classify_mismatches(lock, lock) == []
