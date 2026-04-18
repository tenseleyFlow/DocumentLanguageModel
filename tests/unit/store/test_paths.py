"""Path resolution + layout creation."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.store import errors
from dlm.store.layout import (
    ADAPTER_DIR,
    LOCK_FILENAME,
    LOGS_DIR,
    MANIFEST_FILENAME,
)
from dlm.store.paths import StorePath, dlm_home, ensure_home, for_dlm

VALID_ID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"


class TestDlmHome:
    def test_override_takes_precedence(self, tmp_path: Path) -> None:
        assert dlm_home(override=tmp_path / "custom") == (tmp_path / "custom").resolve()

    def test_env_var_respected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("DLM_HOME", str(tmp_path / "env-home"))
        assert dlm_home() == (tmp_path / "env-home").resolve()

    def test_explicit_override_beats_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("DLM_HOME", str(tmp_path / "env"))
        override = tmp_path / "cli"
        assert dlm_home(override=override) == override.resolve()

    def test_default_on_posix(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("DLM_HOME", raising=False)
        monkeypatch.setattr("dlm.store.paths._current_os_name", lambda: "posix")
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "u")
        assert dlm_home() == tmp_path / "u" / ".dlm"


class TestEnsureHome:
    def test_creates_store_subdir(self, tmp_path: Path) -> None:
        home = ensure_home(override=tmp_path / "h")
        assert home.exists()
        assert (home / "store").exists()


class TestForDlm:
    def test_returns_store_path_under_home(self, tmp_path: Path) -> None:
        sp = for_dlm(VALID_ID, home=tmp_path)
        assert sp.root == tmp_path.resolve() / "store" / VALID_ID

    def test_empty_id_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            for_dlm("", home=tmp_path)


class TestStorePathAccessors:
    @pytest.fixture
    def store(self, tmp_path: Path) -> StorePath:
        return for_dlm(VALID_ID, home=tmp_path)

    def test_manifest_path(self, store: StorePath) -> None:
        assert store.manifest.name == MANIFEST_FILENAME
        assert store.manifest.parent == store.root

    def test_lock_path(self, store: StorePath) -> None:
        assert store.lock.name == LOCK_FILENAME

    def test_adapter_subpaths(self, store: StorePath) -> None:
        assert store.adapter.name == ADAPTER_DIR
        assert store.adapter_versions.parent == store.adapter
        assert store.adapter_version(1).name == "v0001"
        assert store.adapter_version(1234).name == "v1234"

    def test_logs_dir(self, store: StorePath) -> None:
        assert store.logs.name == LOGS_DIR

    def test_adapter_version_zero_rejected(self, store: StorePath) -> None:
        with pytest.raises(ValueError, match="1-indexed"):
            store.adapter_version(0)

    def test_cache_dir_for_slug(self, store: StorePath) -> None:
        assert store.cache_dir_for("qwen2.5-1.5b").name == "qwen2.5-1.5b"

    def test_cache_dir_for_empty_rejected(self, store: StorePath) -> None:
        with pytest.raises(ValueError):
            store.cache_dir_for("")

    def test_export_quant_dir(self, store: StorePath) -> None:
        assert store.export_quant_dir("Q4_K_M").name == "Q4_K_M"

    def test_export_quant_empty_rejected(self, store: StorePath) -> None:
        with pytest.raises(ValueError):
            store.export_quant_dir("")


class TestEnsureLayout:
    @pytest.fixture
    def store(self, tmp_path: Path) -> StorePath:
        sp = for_dlm(VALID_ID, home=tmp_path)
        sp.ensure_layout()
        return sp

    def test_root_created(self, store: StorePath) -> None:
        assert store.root.is_dir()

    def test_always_on_dirs_created(self, store: StorePath) -> None:
        assert store.adapter.is_dir()
        assert store.adapter_versions.is_dir()
        assert store.logs.is_dir()

    def test_lazy_dirs_not_created(self, store: StorePath) -> None:
        # Sprint 08/11/06 own these; Sprint 04 must not materialize them.
        assert not store.replay.exists()
        assert not store.exports.exists()
        assert not store.cache.exists()

    def test_is_idempotent(self, store: StorePath) -> None:
        store.ensure_layout()
        store.ensure_layout()
        assert store.adapter.is_dir()


class TestAdapterCurrentPointer:
    @pytest.fixture
    def store(self, tmp_path: Path) -> StorePath:
        sp = for_dlm(VALID_ID, home=tmp_path)
        sp.ensure_layout()
        return sp

    def test_resolve_returns_none_when_absent(self, store: StorePath) -> None:
        assert store.resolve_current_adapter() is None

    def test_set_and_resolve_roundtrip(self, store: StorePath) -> None:
        version_dir = store.adapter_version(1)
        version_dir.mkdir(parents=True, exist_ok=True)
        store.set_current_adapter(version_dir)
        resolved = store.resolve_current_adapter()
        assert resolved == version_dir.resolve()

    def test_set_rejects_outside_root(self, store: StorePath, tmp_path: Path) -> None:
        rogue = tmp_path / "rogue"
        rogue.mkdir()
        with pytest.raises(ValueError, match="outside store root"):
            store.set_current_adapter(rogue)

    def test_empty_pointer_returns_none(self, store: StorePath) -> None:
        store.adapter_current_pointer.write_text("", encoding="utf-8")
        assert store.resolve_current_adapter() is None

    def test_resolve_rejects_escape_via_parent_refs(self, store: StorePath) -> None:
        # An adversarial pointer pointing outside the root (`..`) must fail.
        store.adapter_current_pointer.write_text("../../escape", encoding="utf-8")
        with pytest.raises(ValueError, match="escapes store root"):
            store.resolve_current_adapter()


class TestErrorsSurface:
    """Sanity-check the errors module exports."""

    def test_unknown_store_error_message(self, tmp_path: Path) -> None:
        err = errors.UnknownStoreError("deadbeef", tmp_path)
        assert "deadbeef" in str(err)
