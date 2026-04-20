"""StoreInspection named-adapter discovery (audit-07 M2)."""

from __future__ import annotations

from pathlib import Path

from dlm.store.inspect import NamedAdapterState, inspect_store
from dlm.store.manifest import Manifest, save_manifest
from dlm.store.paths import StorePath


def _seed_store(tmp_path: Path, dlm_id: str = "01HZ4X7TGZM3J1A2B3C4D5E6FC") -> StorePath:
    store = StorePath(root=tmp_path / "s")
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(dlm_id=dlm_id, base_model="smollm2-135m"),
    )
    return store


class TestFlatStoreHasEmptyNamedAdapters:
    def test_no_named_subdirs_yields_empty_list(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        inspection = inspect_store(store)
        assert inspection.named_adapters == []

    def test_flat_versions_directory_ignored(self, tmp_path: Path) -> None:
        """`adapter/versions/` is the flat layout, not a named adapter."""
        store = _seed_store(tmp_path)
        store.adapter_versions.mkdir(parents=True, exist_ok=True)
        (store.adapter_version(1)).mkdir(parents=True, exist_ok=True)
        inspection = inspect_store(store)
        assert inspection.named_adapters == []


class TestMultiAdapterDiscovery:
    def test_single_named_adapter_surfaces(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        store.ensure_adapter_layout("knowledge")
        v1 = store.adapter_version_for("knowledge", 1)
        v1.mkdir(parents=True)
        store.set_current_adapter_for("knowledge", v1)

        inspection = inspect_store(store)
        assert inspection.named_adapters == [
            NamedAdapterState(name="knowledge", has_current=True, latest_version=1)
        ]

    def test_two_named_adapters_sorted_alphabetically(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        for name, version in [("tone", 2), ("knowledge", 1)]:
            store.ensure_adapter_layout(name)
            for v in range(1, version + 1):
                store.adapter_version_for(name, v).mkdir(parents=True, exist_ok=True)
            store.set_current_adapter_for(name, store.adapter_version_for(name, version))

        inspection = inspect_store(store)
        names = [n.name for n in inspection.named_adapters]
        assert names == ["knowledge", "tone"]

        by_name = {n.name: n for n in inspection.named_adapters}
        assert by_name["knowledge"].latest_version == 1
        assert by_name["tone"].latest_version == 2

    def test_adapter_without_current_pointer(self, tmp_path: Path) -> None:
        """A named adapter dir with versions but no current pointer still surfaces."""
        store = _seed_store(tmp_path)
        store.ensure_adapter_layout("knowledge")
        store.adapter_version_for("knowledge", 1).mkdir(parents=True)
        # Don't set the current.txt pointer.

        inspection = inspect_store(store)
        assert inspection.named_adapters == [
            NamedAdapterState(
                name="knowledge", has_current=False, latest_version=1
            )
        ]

    def test_empty_adapter_dir_without_versions_skipped(self, tmp_path: Path) -> None:
        """A bare `adapter/<name>/` with no `versions/` subdir is skipped
        (not a valid named-adapter shape — could be stray)."""
        store = _seed_store(tmp_path)
        (store.adapter / "ghost").mkdir(parents=True)

        inspection = inspect_store(store)
        assert inspection.named_adapters == []


class TestCoexistenceWithFlatLayout:
    def test_flat_state_preserved_alongside_named(self, tmp_path: Path) -> None:
        """The flat `has_adapter_current` + `adapter_version` are
        independent of the named-adapter list."""
        store = _seed_store(tmp_path)
        # Flat setup.
        flat_v1 = store.adapter_version(1)
        flat_v1.mkdir(parents=True)
        store.set_current_adapter(flat_v1)
        # Named adapter alongside.
        store.ensure_adapter_layout("tone")
        named_v1 = store.adapter_version_for("tone", 1)
        named_v1.mkdir(parents=True)
        store.set_current_adapter_for("tone", named_v1)

        inspection = inspect_store(store)
        assert inspection.has_adapter_current is True
        assert len(inspection.named_adapters) == 1
        assert inspection.named_adapters[0].name == "tone"
