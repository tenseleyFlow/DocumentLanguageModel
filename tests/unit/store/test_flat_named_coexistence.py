"""Flat↔named store layout interaction (audit-08 N10).

A store was used with a flat `training.adapter` doc, then the user
switches to `training.adapters`. The flat `adapter/versions/v0001/`
remains on disk; the new named layouts spin up alongside. These tests
pin:

1. Fresh multi-adapter training on a previously-flat store doesn't
   blow up (paths don't collide).
2. `inspect_store` reports both states sensibly — flat `has_adapter_current`
   reflects the old pointer, `named_adapters` lists the new ones.
3. `resolve_current_adapter` (flat) and `resolve_current_adapter_for(name)`
   (named) are genuinely independent — flipping one doesn't touch the
   other.

Paired with audit-07 M2 (StoreInspection discovery) — this exercises
the "dangling flat state" case M2 opened up.
"""

from __future__ import annotations

from pathlib import Path

from dlm.store.inspect import inspect_store
from dlm.store.manifest import Manifest, save_manifest
from dlm.store.paths import StorePath

_DLM_ID = "01HZ4X7TGZM3J1A2B3C4D5E6FX"


def _seed_store(tmp_path: Path) -> StorePath:
    store = StorePath(root=tmp_path / "coexist")
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(dlm_id=_DLM_ID, base_model="smollm2-135m"),
    )
    return store


def _seed_flat_v1(store: StorePath) -> Path:
    v1 = store.adapter_version(1)
    v1.mkdir(parents=True)
    store.set_current_adapter(v1)
    return v1


def _seed_named_v1(store: StorePath, name: str) -> Path:
    store.ensure_adapter_layout(name)
    v1 = store.adapter_version_for(name, 1)
    v1.mkdir(parents=True)
    store.set_current_adapter_for(name, v1)
    return v1


class TestFlatThenNamed:
    """Flat state lands first, named adapters overlay on top."""

    def test_named_layout_coexists_with_flat_versions_dir(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        flat_v1 = _seed_flat_v1(store)
        # Switch to multi-adapter — ensure_adapter_layout doesn't touch
        # the flat tree.
        named_v1 = _seed_named_v1(store, "knowledge")

        assert flat_v1.is_dir()
        assert named_v1.is_dir()
        assert flat_v1 != named_v1
        # Neither path escapes the other.
        assert str(named_v1).startswith(str(store.adapter / "knowledge"))
        assert "knowledge" not in str(flat_v1)

    def test_flat_pointer_untouched_when_named_pointer_set(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        flat_v1 = _seed_flat_v1(store)
        _seed_named_v1(store, "knowledge")

        # Flat pointer still resolves to its original target.
        assert store.resolve_current_adapter() == flat_v1.resolve()

    def test_named_pointer_untouched_when_flat_pointer_set(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        named_v1 = _seed_named_v1(store, "knowledge")
        _seed_flat_v1(store)

        assert store.resolve_current_adapter_for("knowledge") == named_v1.resolve()


class TestInspectReportsBothStates:
    def test_flat_pointer_plus_named_dir_both_reported(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        _seed_flat_v1(store)
        _seed_named_v1(store, "knowledge")

        inspection = inspect_store(store)
        assert inspection.has_adapter_current is True
        assert inspection.adapter_version == 0  # manifest wasn't updated here
        names = [n.name for n in inspection.named_adapters]
        assert names == ["knowledge"]

    def test_pure_named_store_has_adapter_current_false(self, tmp_path: Path) -> None:
        """A store that only has named adapters (no flat flip) reports
        has_adapter_current=False — the flat `current.txt` isn't set."""
        store = _seed_store(tmp_path)
        _seed_named_v1(store, "knowledge")

        inspection = inspect_store(store)
        assert inspection.has_adapter_current is False
        assert len(inspection.named_adapters) == 1
