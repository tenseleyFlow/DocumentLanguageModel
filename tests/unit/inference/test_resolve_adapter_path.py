"""`resolve_adapter_path` — flat + named layout, missing-adapter errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.inference.errors import AdapterNotFoundError
from dlm.inference.loader import resolve_adapter_path
from dlm.store.paths import StorePath


def _store(tmp_path: Path) -> StorePath:
    s = StorePath(root=tmp_path / "dlm_1")
    s.ensure_layout()
    return s


class TestFlatLayout:
    def test_resolves_current_pointer(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        v1 = s.adapter_version(1)
        v1.mkdir(parents=True)
        s.set_current_adapter(v1)
        assert resolve_adapter_path(s, adapter_name=None) == v1.resolve()

    def test_missing_pointer_raises(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        with pytest.raises(AdapterNotFoundError, match="has `dlm train` run"):
            resolve_adapter_path(s, adapter_name=None)

    def test_pointer_exists_but_dir_missing_raises(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        ghost = s.adapter_version(1)
        ghost.mkdir(parents=True)
        s.set_current_adapter(ghost)
        import shutil

        shutil.rmtree(ghost)
        with pytest.raises(AdapterNotFoundError):
            resolve_adapter_path(s, adapter_name=None)


class TestNamedLayout:
    def test_resolves_named_pointer(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.ensure_adapter_layout("knowledge")
        v1 = s.adapter_version_for("knowledge", 1)
        v1.mkdir(parents=True)
        s.set_current_adapter_for("knowledge", v1)
        assert (
            resolve_adapter_path(s, adapter_name="knowledge") == v1.resolve()
        )

    def test_missing_named_pointer_mentions_adapter_name(
        self, tmp_path: Path
    ) -> None:
        s = _store(tmp_path)
        s.ensure_adapter_layout("knowledge")
        with pytest.raises(AdapterNotFoundError, match="'knowledge'"):
            resolve_adapter_path(s, adapter_name="knowledge")

    def test_named_and_flat_are_independent(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        # Flat is set up.
        flat = s.adapter_version(1)
        flat.mkdir(parents=True)
        s.set_current_adapter(flat)
        # Named pointer doesn't exist.
        with pytest.raises(AdapterNotFoundError):
            resolve_adapter_path(s, adapter_name="knowledge")
        # Flat still resolves.
        assert resolve_adapter_path(s, adapter_name=None) == flat.resolve()
