"""Manifest validation + atomic byte-identical round-trip."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from dlm.store.errors import ManifestCorruptError
from dlm.store.manifest import (
    CURRENT_MANIFEST_SCHEMA_VERSION,
    ExportSummary,
    Manifest,
    TrainingRunSummary,
    load_manifest,
    save_manifest,
    to_canonical_json,
    touch,
)

VALID_ID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"


def _fresh_manifest(**overrides: object) -> Manifest:
    defaults: dict[str, object] = {
        "dlm_id": VALID_ID,
        "base_model": "smollm2-135m",
    }
    defaults.update(overrides)
    return Manifest.model_validate(defaults)


class TestManifestDefaults:
    def test_minimal_manifest(self) -> None:
        m = _fresh_manifest()
        assert m.schema_version == CURRENT_MANIFEST_SCHEMA_VERSION
        assert m.adapter_version == 0
        assert m.training_runs == []
        assert m.exports == []
        assert isinstance(m.created_at, datetime)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Manifest.model_validate(
                {"dlm_id": VALID_ID, "base_model": "x", "unexpected": 1},
            )

    def test_empty_dlm_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Manifest.model_validate({"dlm_id": "", "base_model": "x"})

    def test_negative_adapter_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _fresh_manifest(adapter_version=-1)


class TestNestedTypes:
    def test_training_run_accepts_valid_payload(self) -> None:
        run = TrainingRunSummary(
            run_id=1,
            started_at=datetime(2026, 4, 18),
            adapter_version=1,
            seed=42,
        )
        assert run.status == "completed"

    def test_training_run_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            TrainingRunSummary.model_validate(
                {
                    "run_id": 1,
                    "started_at": datetime(2026, 4, 18, 10, 0),
                    "adapter_version": 1,
                    "seed": 0,
                    "extra": 1,
                }
            )

    def test_export_summary_accepts_valid_payload(self) -> None:
        summary = ExportSummary(exported_at=datetime(2026, 4, 18), quant="Q4_K_M")
        assert summary.merged is False


class TestRoundTrip:
    def test_save_load_byte_identical(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        original = _fresh_manifest(base_model_revision="abc", adapter_version=4)
        save_manifest(path, original)
        first_bytes = path.read_bytes()

        loaded = load_manifest(path)
        save_manifest(path, loaded)
        second_bytes = path.read_bytes()

        assert first_bytes == second_bytes

    def test_loaded_manifest_equals_original(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        m = _fresh_manifest(
            content_hashes={"abc": "def"},
            pinned_versions={"torch": "2.11.0"},
        )
        save_manifest(path, m)
        loaded = load_manifest(path)
        assert loaded == m

    def test_trailing_newline(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        save_manifest(path, _fresh_manifest())
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_sorted_keys(self, tmp_path: Path) -> None:
        text = to_canonical_json(_fresh_manifest())
        # top-level keys appear alphabetically; check a few for sanity.
        idx_adapter = text.index('"adapter_version"')
        idx_base = text.index('"base_model"')
        idx_dlm = text.index('"dlm_id"')
        assert idx_adapter < idx_base < idx_dlm

    def test_atomicity_leaves_no_tmp(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        save_manifest(path, _fresh_manifest())
        siblings = list(tmp_path.iterdir())
        assert siblings == [path], f"tmp files left over: {siblings}"

    def test_source_path_round_trips(self, tmp_path: Path) -> None:
        """Audit-03 minor — Manifest.source_path (Path | None) must
        serialize and round-trip correctly."""
        source = tmp_path / "mydoc.dlm"
        source.touch()  # doesn't need to be valid .dlm for this test
        path = tmp_path / "m.json"
        manifest = _fresh_manifest(source_path=source)
        save_manifest(path, manifest)
        reloaded = load_manifest(path)
        assert reloaded.source_path == source
        # And it's really a Path, not a string.
        assert isinstance(reloaded.source_path, Path)


class TestCorruptHandling:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestCorruptError):
            load_manifest(tmp_path / "absent.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(ManifestCorruptError, match="invalid JSON"):
            load_manifest(path)

    def test_top_level_must_be_object(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1,2,3]", encoding="utf-8")
        with pytest.raises(ManifestCorruptError, match="must be object"):
            load_manifest(path)

    def test_schema_version_mismatch_raises_version_error(self, tmp_path: Path) -> None:
        """Audit-03 — version mismatches surface as the typed subclass so
        Sprint 12b's migrator can catch them specifically, while callers
        that only catch the parent ManifestCorruptError still see them.
        """
        from dlm.store.errors import ManifestVersionError

        path = tmp_path / "old.json"
        path.write_text(
            '{"schema_version": 999, "dlm_id": "x", "base_model": "x"}',
            encoding="utf-8",
        )
        with pytest.raises(ManifestVersionError) as exc_info:
            load_manifest(path)
        assert exc_info.value.found_version == 999
        assert exc_info.value.expected_version == CURRENT_MANIFEST_SCHEMA_VERSION
        # Still catchable as the parent class:
        assert isinstance(exc_info.value, ManifestCorruptError)

    def test_schema_violation_surfaced(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"unknown": true}', encoding="utf-8")
        with pytest.raises(ManifestCorruptError, match="schema"):
            load_manifest(path)


class TestTouch:
    def test_updates_updated_at(self) -> None:
        m = _fresh_manifest()
        original = m.updated_at
        # Small sleep to guarantee a different second.
        import time as _time

        _time.sleep(1.01)
        bumped = touch(m)
        assert bumped.updated_at > original
        assert bumped.dlm_id == m.dlm_id


class TestManifestFrozen:
    def test_mutation_rejected(self) -> None:
        """Audit-03 — Manifest is frozen; use touch() or model_copy()."""
        m = _fresh_manifest()
        with pytest.raises(ValidationError):
            m.adapter_version = 99  # type: ignore[misc]
