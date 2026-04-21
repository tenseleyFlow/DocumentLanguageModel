"""Audio HF-snapshot export (Sprint 35.2 T10) — manifest shape + layout.

Mirrors `test_vl_snapshot.py`. Covers:

- `run_audio_snapshot_export` refuses non-audio specs.
- Writes adapter + manifest + README to the right paths.
- Manifest carries export_target="hf_snapshot" + the base's
  AudioPreprocessorPlan params.
- `verify_artifacts` round-trips.
- `load_audio_snapshot_manifest` deserializes what `_save_manifest` wrote.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.base_models.schema import AudioPreprocessorPlan, BaseModelSpec
from dlm.export.audio_snapshot import (
    AUDIO_SNAPSHOT_SUBDIR,
    SNAPSHOT_MANIFEST_FILENAME,
    SNAPSHOT_README_FILENAME,
    AudioSnapshotManifest,
    load_audio_snapshot_manifest,
    run_audio_snapshot_export,
    verify_artifacts,
)
from dlm.export.errors import ExportError, ExportManifestError
from dlm.store.paths import for_dlm

_VALID_ULID = "01KPMGSTNGSTTSTTSTTSTTSTVA"


def _audio_spec(**overrides: object) -> BaseModelSpec:
    base_kwargs: dict[str, object] = {
        "key": "qwen2-audio-test",
        "hf_id": "Qwen/Qwen2-Audio-test",
        "revision": "c" * 40,
        "architecture": "Qwen2AudioForConditionalGeneration",
        "params": 8_400_000_000,
        "target_modules": ["q_proj"],
        "template": "qwen2-audio",
        "gguf_arch": "qwen2-audio",
        "tokenizer_pre": "qwen2",
        "license_spdx": "Apache-2.0",
        "redistributable": False,
        "size_gb_fp16": 15.0,
        "context_length": 8192,
        "recommended_seq_len": 2048,
        "modality": "audio-language",
        "audio_preprocessor_plan": AudioPreprocessorPlan(
            sample_rate=16_000,
            max_length_seconds=30.0,
            audio_token="<|AUDIO|>",
            num_audio_tokens=750,
        ),
    }
    base_kwargs.update(overrides)
    return BaseModelSpec(**base_kwargs)  # type: ignore[arg-type]


def _text_spec() -> BaseModelSpec:
    return BaseModelSpec(
        key="text-base",
        hf_id="org/text-base",
        revision="a" * 40,
        architecture="LlamaForCausalLM",
        params=1_000_000,
        target_modules=["q_proj"],
        template="chatml",
        gguf_arch="llama",
        tokenizer_pre="llama-bpe",
        license_spdx="Apache-2.0",
        redistributable=True,
        size_gb_fp16=0.5,
        context_length=4096,
        recommended_seq_len=1024,
    )


@pytest.fixture
def populated_store(tmp_path: Path):
    """StorePath with a fake adapter at `adapter/versions/v0001/`."""
    store = for_dlm(_VALID_ULID, home=tmp_path)
    store.ensure_layout()
    v1 = store.adapter_version(1)
    v1.mkdir(parents=True, exist_ok=True)
    (v1 / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (v1 / "adapter_model.safetensors").write_bytes(b"fake audio adapter bytes")
    store.set_current_adapter(v1)
    return store


class TestRefusals:
    def test_text_spec_refused(self, populated_store) -> None:
        with pytest.raises(ExportError, match="only audio-language bases"):
            run_audio_snapshot_export(populated_store, _text_spec())

    def test_missing_adapter_refused(self, tmp_path: Path) -> None:
        store = for_dlm(_VALID_ULID, home=tmp_path)
        store.ensure_layout()
        with pytest.raises(ExportError, match="no current adapter"):
            run_audio_snapshot_export(store, _audio_spec())


class TestSnapshotLayout:
    def test_export_dir_under_exports_hf_audio_snapshot(
        self, populated_store
    ) -> None:
        result = run_audio_snapshot_export(populated_store, _audio_spec())
        assert result.export_dir.name == AUDIO_SNAPSHOT_SUBDIR
        assert result.export_dir.parent == populated_store.exports

    def test_adapter_files_copied(self, populated_store) -> None:
        result = run_audio_snapshot_export(populated_store, _audio_spec())
        assert (result.adapter_dir / "adapter_config.json").exists()
        assert (
            result.adapter_dir / "adapter_model.safetensors"
        ).read_bytes() == b"fake audio adapter bytes"

    def test_manifest_and_readme_written(self, populated_store) -> None:
        result = run_audio_snapshot_export(populated_store, _audio_spec())
        assert result.manifest_path.name == SNAPSHOT_MANIFEST_FILENAME
        assert result.readme_path.name == SNAPSHOT_README_FILENAME
        assert result.manifest_path.exists()
        assert result.readme_path.exists()

    def test_repeat_export_overwrites_adapter(self, populated_store) -> None:
        run_audio_snapshot_export(populated_store, _audio_spec())
        v1 = populated_store.adapter_version(1)
        (v1 / "adapter_model.safetensors").write_bytes(b"new bytes")
        result = run_audio_snapshot_export(populated_store, _audio_spec())
        assert (
            result.adapter_dir / "adapter_model.safetensors"
        ).read_bytes() == b"new bytes"

    def test_vl_and_audio_snapshots_disjoint_subdirs(
        self, populated_store
    ) -> None:
        """Audio + VL snapshots live in different subdirectories.

        A store could in principle hold exports from two separate
        training runs at different times against different bases; the
        subdirectory split keeps them from clobbering each other.
        """
        result = run_audio_snapshot_export(populated_store, _audio_spec())
        assert result.export_dir.name != "hf-snapshot"
        assert result.export_dir.name == AUDIO_SNAPSHOT_SUBDIR


class TestManifestContent:
    def test_export_target_is_hf_snapshot(self, populated_store) -> None:
        run_audio_snapshot_export(populated_store, _audio_spec())
        manifest = load_audio_snapshot_manifest(
            populated_store.exports / AUDIO_SNAPSHOT_SUBDIR
        )
        assert manifest.export_target == "hf_snapshot"
        assert manifest.modality == "audio-language"

    def test_base_pinned_in_manifest(self, populated_store) -> None:
        run_audio_snapshot_export(populated_store, _audio_spec())
        manifest = load_audio_snapshot_manifest(
            populated_store.exports / AUDIO_SNAPSHOT_SUBDIR
        )
        assert manifest.base_model_hf_id == "Qwen/Qwen2-Audio-test"
        assert manifest.base_model_revision == "c" * 40
        assert manifest.base_model_architecture == "Qwen2AudioForConditionalGeneration"

    def test_preprocessor_params_pinned(self, populated_store) -> None:
        run_audio_snapshot_export(populated_store, _audio_spec())
        manifest = load_audio_snapshot_manifest(
            populated_store.exports / AUDIO_SNAPSHOT_SUBDIR
        )
        assert manifest.audio_token == "<|AUDIO|>"
        assert manifest.num_audio_tokens == 750
        assert manifest.sample_rate == 16_000
        assert manifest.max_length_seconds == 30.0

    def test_adapter_version_recorded(self, populated_store) -> None:
        run_audio_snapshot_export(populated_store, _audio_spec())
        manifest = load_audio_snapshot_manifest(
            populated_store.exports / AUDIO_SNAPSHOT_SUBDIR
        )
        assert manifest.adapter_version == 1

    def test_adapter_artifacts_listed(self, populated_store) -> None:
        run_audio_snapshot_export(populated_store, _audio_spec())
        manifest = load_audio_snapshot_manifest(
            populated_store.exports / AUDIO_SNAPSHOT_SUBDIR
        )
        paths = {entry.path for entry in manifest.artifacts}
        assert "adapter/adapter_config.json" in paths
        assert "adapter/adapter_model.safetensors" in paths


class TestVerifyArtifacts:
    def test_pristine_snapshot_verifies(self, populated_store) -> None:
        run_audio_snapshot_export(populated_store, _audio_spec())
        export_dir = populated_store.exports / AUDIO_SNAPSHOT_SUBDIR
        manifest = load_audio_snapshot_manifest(export_dir)
        verify_artifacts(export_dir, manifest)  # no raise

    def test_tampered_artifact_detected(self, populated_store) -> None:
        run_audio_snapshot_export(populated_store, _audio_spec())
        export_dir = populated_store.exports / AUDIO_SNAPSHOT_SUBDIR
        manifest = load_audio_snapshot_manifest(export_dir)
        target = export_dir / manifest.artifacts[0].path
        target.write_bytes(b"tampered")
        with pytest.raises(ExportManifestError, match="sha256 mismatch"):
            verify_artifacts(export_dir, manifest)

    def test_missing_artifact_detected(self, populated_store) -> None:
        run_audio_snapshot_export(populated_store, _audio_spec())
        export_dir = populated_store.exports / AUDIO_SNAPSHOT_SUBDIR
        manifest = load_audio_snapshot_manifest(export_dir)
        (export_dir / manifest.artifacts[0].path).unlink()
        with pytest.raises(ExportManifestError, match="missing declared artifact"):
            verify_artifacts(export_dir, manifest)


class TestManifestLoadFailures:
    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ExportManifestError, match="missing"):
            load_audio_snapshot_manifest(tmp_path)

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / SNAPSHOT_MANIFEST_FILENAME).write_text("not json", encoding="utf-8")
        with pytest.raises(ExportManifestError, match="cannot parse"):
            load_audio_snapshot_manifest(tmp_path)


class TestManifestModelDirect:
    def test_frozen(self) -> None:
        from datetime import UTC, datetime

        manifest = AudioSnapshotManifest(
            created_at=datetime.now(UTC).replace(tzinfo=None),
            created_by="dlm-test",
            base_model_hf_id="x/y",
            base_model_revision="a" * 40,
            base_model_architecture="X",
            audio_token="<|AUDIO|>",
            num_audio_tokens=750,
            sample_rate=16_000,
            max_length_seconds=30.0,
            adapter_version=1,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            manifest.adapter_version = 2  # type: ignore[misc]


class TestReadmeContent:
    def test_mentions_architecture_class(self, populated_store) -> None:
        """README should name the architecture class for load snippet."""
        result = run_audio_snapshot_export(populated_store, _audio_spec())
        body = result.readme_path.read_text(encoding="utf-8")
        assert "Qwen2AudioForConditionalGeneration" in body
        # Sample-rate + placeholder token surface the runtime contract.
        assert "16000 Hz" in body
        assert "<|AUDIO|>" in body
