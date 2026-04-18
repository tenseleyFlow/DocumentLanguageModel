"""Verify the golden-output registry end-to-end."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures import golden


@pytest.fixture(autouse=True)
def _isolated_golden_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect the golden root so tests don't write into the repo tree."""
    monkeypatch.setattr(golden, "GOLDEN_ROOT", tmp_path / "golden")


class TestGoldenPath:
    def test_includes_torch_version(self) -> None:
        p = golden.golden_path("demo", torch_version="2.11.0+cpu")
        assert "demo" in p.name
        assert "torch-2.11.0_cpu" in p.name


class TestAssertGolden:
    def test_missing_golden_raises(self) -> None:
        with pytest.raises(golden.MissingGoldenError):
            golden.assert_golden({"x": 1}, "absent", torch_version="0.0.0")

    def test_update_then_assert_roundtrips(self) -> None:
        value = {"loss": [1.0, 0.5, 0.25], "steps": 3}
        golden.assert_golden(value, "run-a", torch_version="1.2.3", update=True)
        # Second call without update should succeed.
        golden.assert_golden(value, "run-a", torch_version="1.2.3")

    def test_mismatch_raises_with_expected_and_actual(self) -> None:
        golden.assert_golden({"x": 1}, "run-b", torch_version="1.2.3", update=True)
        with pytest.raises(AssertionError) as excinfo:
            golden.assert_golden({"x": 2}, "run-b", torch_version="1.2.3")
        msg = str(excinfo.value)
        assert "Expected" in msg
        assert "Actual" in msg

    def test_torch_version_keys_separate_goldens(self) -> None:
        golden.assert_golden({"v": "old"}, "drift", torch_version="2.0.0", update=True)
        golden.assert_golden({"v": "new"}, "drift", torch_version="2.11.0", update=True)
        # Both must still pass under their own keys.
        golden.assert_golden({"v": "old"}, "drift", torch_version="2.0.0")
        golden.assert_golden({"v": "new"}, "drift", torch_version="2.11.0")

    def test_canonical_comparison_ignores_key_order(self) -> None:
        golden.assert_golden({"a": 1, "b": 2}, "ordered", torch_version="x", update=True)
        # Different insertion order must still match.
        golden.assert_golden({"b": 2, "a": 1}, "ordered", torch_version="x")
