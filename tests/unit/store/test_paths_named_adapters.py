"""Path resolver coverage for the named-adapter layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.store.paths import StorePath


def _store(tmp_path: Path) -> StorePath:
    store = StorePath(root=tmp_path / "dlm_1")
    store.ensure_layout()
    return store


class TestNamedAdapterPaths:
    def test_adapter_dir_for_name(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert s.adapter_dir_for("knowledge") == s.adapter / "knowledge"

    def test_adapter_versions_for_name(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert s.adapter_versions_for("tone") == s.adapter / "tone" / "versions"

    def test_adapter_version_for_name_pads_four_digits(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert (
            s.adapter_version_for("tone", 7)
            == s.adapter / "tone" / "versions" / "v0007"
        )

    def test_pointer_path_for_name(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert (
            s.adapter_current_pointer_for("knowledge")
            == s.adapter / "knowledge" / "current.txt"
        )


class TestNamedAdapterValidation:
    @pytest.mark.parametrize(
        "bad",
        ["", "Knowledge", "1tone", "tone-name", "tone.name", "../escape", "_x"],
    )
    def test_invalid_names_rejected(self, tmp_path: Path, bad: str) -> None:
        s = _store(tmp_path)
        with pytest.raises(ValueError, match="not valid"):
            s.adapter_dir_for(bad)

    def test_version_must_be_positive(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        with pytest.raises(ValueError, match="1-indexed"):
            s.adapter_version_for("knowledge", 0)


class TestEnsureAdapterLayout:
    def test_creates_versions_dir(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.ensure_adapter_layout("knowledge")
        assert (s.adapter / "knowledge" / "versions").is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.ensure_adapter_layout("tone")
        s.ensure_adapter_layout("tone")  # no-op; no error
        assert (s.adapter / "tone" / "versions").is_dir()

    def test_multiple_names_coexist(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.ensure_adapter_layout("knowledge")
        s.ensure_adapter_layout("tone")
        assert (s.adapter / "knowledge" / "versions").is_dir()
        assert (s.adapter / "tone" / "versions").is_dir()


class TestPointerRoundTrip:
    def test_set_and_resolve(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.ensure_adapter_layout("knowledge")
        v1 = s.adapter_version_for("knowledge", 1)
        v1.mkdir(parents=True)
        s.set_current_adapter_for("knowledge", v1)
        assert s.resolve_current_adapter_for("knowledge") == v1.resolve()

    def test_resolve_missing_pointer_returns_none(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.ensure_adapter_layout("knowledge")
        assert s.resolve_current_adapter_for("knowledge") is None

    def test_resolve_empty_pointer_returns_none(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.ensure_adapter_layout("knowledge")
        s.adapter_current_pointer_for("knowledge").write_text("\n")
        assert s.resolve_current_adapter_for("knowledge") is None

    def test_set_rejects_path_outside_store(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        outside = tmp_path / "elsewhere" / "v0001"
        outside.mkdir(parents=True)
        with pytest.raises(ValueError, match="outside store root"):
            s.set_current_adapter_for("knowledge", outside)

    def test_flat_and_named_pointers_are_independent(self, tmp_path: Path) -> None:
        """The flat `adapter/current.txt` and named pointers are
        separate files — flipping one must not affect the other."""
        s = _store(tmp_path)
        s.adapter_versions.mkdir(parents=True, exist_ok=True)
        flat_v = s.adapter_version(1)
        flat_v.mkdir(parents=True)
        s.set_current_adapter(flat_v)

        s.ensure_adapter_layout("knowledge")
        named_v = s.adapter_version_for("knowledge", 1)
        named_v.mkdir(parents=True)
        s.set_current_adapter_for("knowledge", named_v)

        assert s.resolve_current_adapter() == flat_v.resolve()
        assert s.resolve_current_adapter_for("knowledge") == named_v.resolve()
