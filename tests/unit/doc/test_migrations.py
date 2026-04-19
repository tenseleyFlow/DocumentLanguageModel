"""Migration registry + dispatcher + enforcement test (Sprint 12b)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from dlm.doc.errors import UnsupportedMigrationError
from dlm.doc.migrations import MIGRATORS, register
from dlm.doc.migrations.dispatch import apply_pending
from dlm.doc.schema import CURRENT_SCHEMA_VERSION


@pytest.fixture
def scratch_registry() -> Iterator[None]:
    """Snapshot + restore MIGRATORS around tests that register migrators."""
    saved = dict(MIGRATORS)
    try:
        MIGRATORS.clear()
        yield
    finally:
        MIGRATORS.clear()
        MIGRATORS.update(saved)


class TestRegister:
    def test_duplicate_registration_raises(self, scratch_registry: None) -> None:
        @register(from_version=1)
        def _a(raw: dict[str, object]) -> dict[str, object]:
            return raw

        with pytest.raises(AssertionError, match="duplicate migrator"):

            @register(from_version=1)
            def _b(raw: dict[str, object]) -> dict[str, object]:
                return raw

    def test_registered_callable_returned(self, scratch_registry: None) -> None:
        @register(from_version=42)
        def _fn(raw: dict[str, object]) -> dict[str, object]:
            return raw

        assert MIGRATORS[42] is _fn


class TestApplyPending:
    def test_already_current_is_noop(self, scratch_registry: None) -> None:
        raw = {"dlm_version": 3, "x": 1}
        migrated, applied = apply_pending(raw, target_version=3)
        assert migrated == raw
        assert applied == []

    def test_higher_than_target_is_noop(self, scratch_registry: None) -> None:
        """Sprint 14+ users who bumped ahead don't get downgraded."""
        raw = {"dlm_version": 5}
        migrated, applied = apply_pending(raw, target_version=3)
        assert migrated == raw
        assert applied == []

    def test_chain_runs_in_order(self, scratch_registry: None) -> None:
        @register(from_version=1)
        def _v1(raw: dict[str, object]) -> dict[str, object]:
            return {**raw, "added_in_v2": True}

        @register(from_version=2)
        def _v2(raw: dict[str, object]) -> dict[str, object]:
            return {**raw, "added_in_v3": True}

        raw = {"dlm_version": 1, "base": "x"}
        migrated, applied = apply_pending(raw, target_version=3)
        assert applied == [1, 2]
        assert migrated["dlm_version"] == 3
        assert migrated["added_in_v2"] is True
        assert migrated["added_in_v3"] is True

    def test_default_version_is_one(self, scratch_registry: None) -> None:
        """Raw dicts without explicit dlm_version start at v1."""

        @register(from_version=1)
        def _v1(raw: dict[str, object]) -> dict[str, object]:
            return {**raw, "migrated": True}

        migrated, applied = apply_pending({"base": "x"}, target_version=2)
        assert applied == [1]
        assert migrated["dlm_version"] == 2
        assert migrated["migrated"] is True

    def test_missing_migrator_raises(self, scratch_registry: None) -> None:
        with pytest.raises(UnsupportedMigrationError, match="no migrator"):
            apply_pending({"dlm_version": 1}, target_version=2)

    def test_non_int_version_raises(self, scratch_registry: None) -> None:
        with pytest.raises(UnsupportedMigrationError, match="must be int"):
            apply_pending({"dlm_version": "1"}, target_version=2)

    def test_input_not_mutated(self, scratch_registry: None) -> None:
        @register(from_version=1)
        def _v1(raw: dict[str, object]) -> dict[str, object]:
            return {**raw, "new": 1}

        raw = {"dlm_version": 1, "base": "x"}
        apply_pending(raw, target_version=2)
        assert raw == {"dlm_version": 1, "base": "x"}


class TestCoverageEnforcement:
    """Block a PR that bumps CURRENT_SCHEMA_VERSION without a matching migrator.

    This test is the whole point of the framework: every intermediate
    version from 1 to CURRENT-1 must have a migrator registered. At
    launch (CURRENT=1), the expected set is empty — registering one at
    that point would indicate a forgotten version bump.
    """

    def test_migrators_span_required_range(self) -> None:
        expected = set(range(1, CURRENT_SCHEMA_VERSION))
        assert set(MIGRATORS.keys()) == expected, (
            f"MIGRATORS keys {sorted(MIGRATORS)!r} do not match expected "
            f"range [1, {CURRENT_SCHEMA_VERSION}). If you bumped "
            "CURRENT_SCHEMA_VERSION, register a migrator under "
            "src/dlm/doc/migrations/."
        )
