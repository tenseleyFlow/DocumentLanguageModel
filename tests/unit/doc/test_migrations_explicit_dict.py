"""Sprint 39 m8 — shipped doc migrators are declared explicitly."""

from __future__ import annotations

from dlm.doc.migrations import (
    MIGRATORS,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    v8,
    v9,
    v10,
    v11,
)


def test_explicit_registry_points_at_shipped_modules() -> None:
    assert {
        1: v1.migrate,
        2: v2.migrate,
        3: v3.migrate,
        4: v4.migrate,
        5: v5.migrate,
        6: v6.migrate,
        7: v7.migrate,
        8: v8.migrate,
        9: v9.migrate,
        10: v10.migrate,
        11: v11.migrate,
    } == MIGRATORS
