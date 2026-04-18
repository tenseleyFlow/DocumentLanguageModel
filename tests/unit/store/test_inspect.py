"""Store inspection + orphan detection."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from dlm.store.inspect import inspect_store
from dlm.store.manifest import Manifest, TrainingRunSummary, save_manifest
from dlm.store.paths import StorePath, for_dlm
from tests.fixtures.dlm_factory import make_dlm

VALID_ID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"
OTHER_ID = "01HZ4X7TGZM3J1A2B3C4D5E6F8"


@pytest.fixture
def populated_store(tmp_path: Path) -> StorePath:
    """A store with manifest + empty layout so inspect_store can walk it."""
    store = for_dlm(VALID_ID, home=tmp_path)
    store.ensure_layout()
    manifest = Manifest(
        dlm_id=VALID_ID,
        base_model="smollm2-135m",
        base_model_revision="abc",
        adapter_version=2,
        training_runs=[
            TrainingRunSummary(
                run_id=1,
                started_at=datetime(2026, 4, 18, 10, 0),
                ended_at=datetime(2026, 4, 18, 10, 5),
                adapter_version=1,
                seed=42,
                steps=100,
            ),
            TrainingRunSummary(
                run_id=2,
                started_at=datetime(2026, 4, 18, 11, 0),
                ended_at=datetime(2026, 4, 18, 11, 10),
                adapter_version=2,
                seed=42,
                steps=150,
            ),
        ],
        content_hashes={"abc": "def"},
        pinned_versions={"torch": "2.11.0"},
    )
    save_manifest(store.manifest, manifest)
    # Populate an adapter version dir + pointer to simulate Sprint 09 output.
    v2 = store.adapter_version(2)
    v2.mkdir(parents=True, exist_ok=True)
    (v2 / "adapter_model.safetensors").write_bytes(b"\x00" * 1024)
    store.set_current_adapter(v2)
    return store


class TestInspectBasic:
    def test_reports_manifest_fields(self, populated_store: StorePath) -> None:
        result = inspect_store(populated_store)
        assert result.dlm_id == VALID_ID
        assert result.base_model == "smollm2-135m"
        assert result.base_model_revision == "abc"
        assert result.adapter_version == 2
        assert result.training_runs == 2

    def test_has_adapter_current_true(self, populated_store: StorePath) -> None:
        result = inspect_store(populated_store)
        assert result.has_adapter_current is True

    def test_last_trained_at_is_max(self, populated_store: StorePath) -> None:
        result = inspect_store(populated_store)
        assert result.last_trained_at == datetime(2026, 4, 18, 11, 10)

    def test_total_size_includes_adapter_bytes(self, populated_store: StorePath) -> None:
        result = inspect_store(populated_store)
        assert result.total_size_bytes >= 1024

    def test_replay_size_is_zero_when_empty(self, populated_store: StorePath) -> None:
        result = inspect_store(populated_store)
        assert result.replay_size_bytes == 0

    def test_content_hashes_and_pins_propagate(self, populated_store: StorePath) -> None:
        result = inspect_store(populated_store)
        assert result.content_hashes == {"abc": "def"}
        assert result.pinned_versions == {"torch": "2.11.0"}


class TestInspectNoAdapter:
    def test_has_adapter_current_false_when_not_set(self, tmp_path: Path) -> None:
        store = for_dlm(VALID_ID, home=tmp_path)
        store.ensure_layout()
        manifest = Manifest(dlm_id=VALID_ID, base_model="x")
        save_manifest(store.manifest, manifest)
        result = inspect_store(store)
        assert result.has_adapter_current is False
        assert result.last_trained_at is None
        assert result.training_runs == 0


class TestOrphanDetection:
    def test_orphan_when_source_missing(self, populated_store: StorePath, tmp_path: Path) -> None:
        missing = tmp_path / "gone.dlm"
        result = inspect_store(populated_store, source_path=missing)
        assert result.orphaned is True
        assert result.source_path == missing

    def test_not_orphan_when_source_exists_and_matches(
        self, populated_store: StorePath, tmp_path: Path
    ) -> None:
        source = tmp_path / "mine.dlm"
        source.write_text(make_dlm(dlm_id=VALID_ID), encoding="utf-8")
        result = inspect_store(populated_store, source_path=source)
        assert result.orphaned is False

    def test_orphan_when_source_has_wrong_dlm_id(
        self, populated_store: StorePath, tmp_path: Path
    ) -> None:
        source = tmp_path / "other.dlm"
        source.write_text(make_dlm(dlm_id=OTHER_ID), encoding="utf-8")
        result = inspect_store(populated_store, source_path=source)
        assert result.orphaned is True

    def test_not_orphan_when_source_path_unknown(self, populated_store: StorePath) -> None:
        result = inspect_store(populated_store)
        assert result.orphaned is False
        assert result.source_path is None

    def test_manifest_source_path_used_when_no_arg(self, tmp_path: Path) -> None:
        source = tmp_path / "mine.dlm"
        source.write_text(make_dlm(dlm_id=VALID_ID), encoding="utf-8")
        store = for_dlm(VALID_ID, home=tmp_path)
        store.ensure_layout()
        save_manifest(
            store.manifest,
            Manifest(
                dlm_id=VALID_ID,
                base_model="x",
                source_path=source,
            ),
        )
        result = inspect_store(store)
        assert result.orphaned is False
        assert result.source_path == source

    def test_corrupt_source_file_treated_as_orphan(
        self, populated_store: StorePath, tmp_path: Path
    ) -> None:
        source = tmp_path / "garbage.dlm"
        source.write_text("not a dlm", encoding="utf-8")
        result = inspect_store(populated_store, source_path=source)
        assert result.orphaned is True


class TestTimelineEdges:
    def test_running_run_does_not_set_last_trained_at(self, tmp_path: Path) -> None:
        store = for_dlm(VALID_ID, home=tmp_path)
        store.ensure_layout()
        manifest = Manifest(
            dlm_id=VALID_ID,
            base_model="x",
            training_runs=[
                TrainingRunSummary(
                    run_id=1,
                    started_at=datetime(2026, 4, 18),
                    ended_at=None,
                    status="running",
                    adapter_version=1,
                    seed=0,
                )
            ],
        )
        save_manifest(store.manifest, manifest)
        result = inspect_store(store)
        assert result.last_trained_at is None

    def test_mixed_runs_pick_latest_ended(self, tmp_path: Path) -> None:
        store = for_dlm(VALID_ID, home=tmp_path)
        store.ensure_layout()
        base = datetime(2026, 4, 18, 10, 0)
        manifest = Manifest(
            dlm_id=VALID_ID,
            base_model="x",
            training_runs=[
                TrainingRunSummary(
                    run_id=1,
                    started_at=base,
                    ended_at=base + timedelta(minutes=5),
                    adapter_version=1,
                    seed=0,
                ),
                TrainingRunSummary(
                    run_id=2,
                    started_at=base + timedelta(minutes=10),
                    ended_at=None,
                    status="running",
                    adapter_version=2,
                    seed=0,
                ),
            ],
        )
        save_manifest(store.manifest, manifest)
        result = inspect_store(store)
        assert result.last_trained_at == base + timedelta(minutes=5)
