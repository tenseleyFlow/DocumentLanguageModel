"""Pack-migration registry + dispatcher + coverage enforcement (Sprint 14)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from dlm.pack.errors import PackFormatVersionError
from dlm.pack.format import CURRENT_PACK_FORMAT_VERSION
from dlm.pack.migrations import PACK_MIGRATORS, register
from dlm.pack.migrations.dispatch import apply_pending
from dlm.pack.migrations.v1 import migrate as migrate_v1


@pytest.fixture
def scratch_registry() -> Iterator[None]:
    """Snapshot + restore PACK_MIGRATORS around mutative tests."""
    saved = dict(PACK_MIGRATORS)
    try:
        PACK_MIGRATORS.clear()
        yield
    finally:
        PACK_MIGRATORS.clear()
        PACK_MIGRATORS.update(saved)


class TestRegister:
    def test_duplicate_raises(self, scratch_registry: None) -> None:
        @register(from_version=1)
        def _a(root: Path) -> Path:
            return root

        with pytest.raises(AssertionError, match="duplicate pack migrator"):

            @register(from_version=1)
            def _b(root: Path) -> Path:
                return root


class TestApplyPending:
    def test_already_current_is_noop(self, scratch_registry: None, tmp_path: Path) -> None:
        root, applied = apply_pending(tmp_path, from_version=CURRENT_PACK_FORMAT_VERSION)
        assert root == tmp_path
        assert applied == []

    def test_newer_than_current_raises(self, scratch_registry: None, tmp_path: Path) -> None:
        with pytest.raises(PackFormatVersionError):
            apply_pending(tmp_path, from_version=CURRENT_PACK_FORMAT_VERSION + 1)

    def test_chain_runs(self, scratch_registry: None, tmp_path: Path) -> None:
        calls: list[int] = []

        @register(from_version=1)
        def _v1_to_v2(root: Path) -> Path:
            calls.append(1)
            return root

        @register(from_version=2)
        def _v2_to_v3(root: Path) -> Path:
            calls.append(2)
            return root

        # Simulate CURRENT=3 by targeting directly.
        # (We can't mutate CURRENT safely; just verify the chain with
        # a manual loop equivalent.)
        # Use apply_pending with a temporary bump via monkeypatch.
        import dlm.pack.migrations.dispatch as dispatch_module

        original = dispatch_module.CURRENT_PACK_FORMAT_VERSION
        dispatch_module.CURRENT_PACK_FORMAT_VERSION = 3
        try:
            apply_pending(tmp_path, from_version=1)
        finally:
            dispatch_module.CURRENT_PACK_FORMAT_VERSION = original
        assert calls == [1, 2]

    def test_gap_in_registry_raises(self, scratch_registry: None, tmp_path: Path) -> None:
        # CURRENT=2 but no v1 migrator registered → gap.
        import dlm.pack.migrations.dispatch as dispatch_module

        original = dispatch_module.CURRENT_PACK_FORMAT_VERSION
        dispatch_module.CURRENT_PACK_FORMAT_VERSION = 2
        try:
            with pytest.raises(PackFormatVersionError):
                apply_pending(tmp_path, from_version=1)
        finally:
            dispatch_module.CURRENT_PACK_FORMAT_VERSION = original


class TestCoverageEnforcement:
    """A PR that bumps `CURRENT_PACK_FORMAT_VERSION` without registering a
    migrator fails this test. At v1 the expected range is empty."""

    def test_migrators_span_required_range(self) -> None:
        expected = set(range(1, CURRENT_PACK_FORMAT_VERSION))
        assert set(PACK_MIGRATORS.keys()) == expected, (
            f"PACK_MIGRATORS keys {sorted(PACK_MIGRATORS)!r} do not match "
            f"expected range [1, {CURRENT_PACK_FORMAT_VERSION}). Register a "
            "migrator under src/dlm/pack/migrations/ when bumping the version."
        )


class TestV1IdentityMigrator:
    def test_v1_identity_migrator_returns_same_root(self, tmp_path: Path) -> None:
        assert migrate_v1(tmp_path) == tmp_path
