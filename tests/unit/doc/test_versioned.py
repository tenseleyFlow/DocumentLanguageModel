"""validate_versioned — migration-aware frontmatter validation (Sprint 12b)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from dlm.doc import versioned as versioned_module
from dlm.doc.errors import (
    DlmVersionError,
    SchemaValidationError,
    UnsupportedMigrationError,
)
from dlm.doc.migrations import MIGRATORS, register
from dlm.doc.schema import CURRENT_SCHEMA_VERSION
from dlm.doc.versioned import validate_versioned

_VALID_ULID = "01JQ7Z0000000000000000000A"


@pytest.fixture
def scratch_registry() -> Iterator[None]:
    saved = dict(MIGRATORS)
    try:
        MIGRATORS.clear()
        yield
    finally:
        MIGRATORS.clear()
        MIGRATORS.update(saved)


class TestCurrent:
    def test_v1_validates_cleanly(self) -> None:
        fm = validate_versioned({"dlm_id": _VALID_ULID, "base_model": "smollm2-135m"})
        assert fm.dlm_version == CURRENT_SCHEMA_VERSION
        assert fm.dlm_id == _VALID_ULID

    def test_pydantic_errors_surface_as_schema_error(self) -> None:
        with pytest.raises(SchemaValidationError):
            validate_versioned({"dlm_id": "not-a-ulid", "base_model": "smollm2-135m"})


class TestFutureVersion:
    def test_newer_than_current_raises(self) -> None:
        with pytest.raises(DlmVersionError, match="newer than this parser"):
            validate_versioned(
                {
                    "dlm_id": _VALID_ULID,
                    "base_model": "smollm2-135m",
                    "dlm_version": CURRENT_SCHEMA_VERSION + 100,
                }
            )


class TestMalformedVersionField:
    """`dlm_version` must be an `int` — strings/floats/bools get caught early."""

    @pytest.mark.parametrize(
        "bad",
        ["1", "5", 2.0, True, False, None, [1]],
    )
    def test_non_int_version_raises_schema_error(self, bad: object) -> None:
        with pytest.raises(SchemaValidationError, match="dlm_version must be an integer"):
            validate_versioned(
                {
                    "dlm_id": _VALID_ULID,
                    "base_model": "smollm2-135m",
                    "dlm_version": bad,
                }
            )


class TestMigratedPath:
    def test_migration_applied_before_pydantic(self, scratch_registry: None) -> None:
        """A registered migrator rewrites the dict before validation."""

        original = versioned_module.CURRENT_SCHEMA_VERSION

        @register(from_version=original)
        def _pre_current(raw: dict[str, object]) -> dict[str, object]:
            # Simulated migration: drop an obsolete field the sub-current doc had.
            return {k: v for k, v in raw.items() if k != "legacy_field"}

        versioned_module.CURRENT_SCHEMA_VERSION = original + 1
        try:
            fm = validate_versioned(
                {
                    "dlm_id": _VALID_ULID,
                    "base_model": "smollm2-135m",
                    "dlm_version": original,
                    "legacy_field": "would-fail-extra-forbid",
                }
            )
            assert fm.dlm_version == original + 1
        finally:
            versioned_module.CURRENT_SCHEMA_VERSION = original

    def test_missing_migrator_raises(self, scratch_registry: None) -> None:
        """Sub-current doc + empty registry → UnsupportedMigrationError."""
        original = versioned_module.CURRENT_SCHEMA_VERSION
        versioned_module.CURRENT_SCHEMA_VERSION = original + 1
        try:
            with pytest.raises(UnsupportedMigrationError):
                validate_versioned(
                    {
                        "dlm_id": _VALID_ULID,
                        "base_model": "smollm2-135m",
                        "dlm_version": 1,
                    }
                )
        finally:
            versioned_module.CURRENT_SCHEMA_VERSION = original
